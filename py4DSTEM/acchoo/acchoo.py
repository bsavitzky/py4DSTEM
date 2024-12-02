import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi
from scipy.ndimage import binary_opening, binary_closing
from py4DSTEM.utils import get_maxima_2D, get_voronoi_vertices
from py4DSTEM.visualize import show, show_points, add_vector

class ACCHOO:
    """
    implements the ACCHOO algorithm. usage:

    import ACCHOO
    a = ACCHOO(disks)             # type=BraggVectors
    a.get_kpoints(...)
    a.show_voronoi(...)
    a.label_empty_pixels(...)
    a.clean_empty_pixels(...)
    a.run()
    a.show_result()

    Label meanings are:
    0 = unlabelled
    1 = labelled by current path
    2 = labelled
    3 = change from current path
    4 = change from a path
    """
    def __init__(
        self,
        disks,
        upsample=1,
        min_points_empty=3,
        min_points=2,
        min_inten_empty=0,
        min_inten=0,
        dist_frac_tol=1,
        numb_frac_tol=0,
        seed_picker='max',
        num_lowq=3,
        num_highi=1,
        a_scale=1,
        store_xtalforms=True,
        ):
        """
        Parameters
        ----------
        disks : BraggVectors
        upsample : integer
            upsample factor for the Bragg vector histogram
        min_points_empty : integer
            threshold # data points for 'empty' pixels during initialization
        min_points : integer
            threshold # unmasked data points for 'empty' pixels during run
        min_inten_empty : float
            threshold intensity for data points to be counted as 'empty' or not
        min_inten : bool
            data inclusion intensity threshold
        dist_frac_tol : number
            data points which are just over a voronoi ridge from a masked to an
            unmasked channel may be ignored if the distance fraction
            (dist_to_masked_vor_point / dis_to_unmasked_vor_point) is under this
            threshold tolerance. 1 (default) is no tolerance; 1.05 means if the
            distance to the closest masked seed is within 5% of the closest
            unmasked seed distance, the point is disregarded
        numb_frac_tol : number
            unmasked data points will be ignored unless their fraction of the total
            number of points is greater than numb_frac_tol
        seed_picker : str in 'max' or 'most' or 'random' or 'front' or 'back'
            strategy for choosing crystal seed points
        num_lowq : int
            when finding basis vector options, consider the num_lowq data points
            closest to the origin
        num_highi : int
            when finding basis vector options, consider the num_highi brightest
            data points
        a_scale : number
            scale `a` in the cost function (missed data point weight)
        store_xtalforms : bool
            toggles storing an array describing for each path's coverage
        """
        self._setup_disks(disks,upsample)
        self._reset_path_vars()
        self._setup()
        self.set_min_points_empty(min_points_empty)
        self.set_min_points(min_points)
        self.set_min_inten_empty(min_inten_empty)
        self.set_min_inten(min_inten)
        self.set_dist_frac_tol(dist_frac_tol)
        self.set_numb_frac_tol(numb_frac_tol)
        self.set_seed_picker(seed_picker)
        self.set_num_lowq(num_lowq)
        self.set_num_highi(num_highi)
        self.set_a_scale(a_scale)
        self.set_store_xtalforms(store_xtalforms)

    def _setup_disks(self, disks, upsample=1):
        self.d = disks
        self.upsample = upsample
        self.bvm = disks.histogram(sampling=upsample)

    def _reset_path_vars(self):
        self._pos = 0
        self._dirs = []
        self._coords = []

    def _setup(self):
        s = self.shape
        self.labels = np.zeros(s,dtype=int)
        self.empty = np.zeros(s,dtype=bool)
        self.state_crystals = [[[] for y in range(s[1])] for x in range(s[0])]
        self.state_masks = [[[] for y in range(s[1])] for x in range(s[0])]
        self.seeds = []
        self.crystals = []
        self.crystal_seeds = []
        self.crystalforms = []
        self._scores=np.ones(s,dtype=float)
        self._score_caps=np.ones(s,dtype=int)
        self._labelled_empties = False

    # convenience properties and setters
    @property
    def N(self):
        return len(self._voronoi.points)
    @property
    def qpixsize(self):
        return self.d.calibration.get_Q_pixel_size()/self.upsample
    @property
    def shape(self):
        return self.d.shape

    # param setters
    def set_min_points_empty(self,min_points_empty):
        self.min_empty = min_points_empty
    def set_min_points(self,min_points):
        self.min = min_points
    def set_min_inten_empty(self,min_inten_empty):
        self.thresh_empty = min_inten_empty
    def set_min_inten(self,min_inten):
        self.thresh = min_inten
    def set_dist_frac_tol(self,dist_frac_tol):
        self.distance_frac_tolerance = dist_frac_tol
    def set_numb_frac_tol(self,numb_frac_tol):
        self.number_frac_tolerance = numb_frac_tol
    def set_seed_picker(self,seed_picker):
        assert(seed_picker in ['max','most','random','front','back',]), f"Unknown value for seed picker {seed_picker}!"
        self.seed_picker = seed_picker
        # for "maximum" picker, perform calc upfront
        if seed_picker=='max':
            self._inten_tot = np.zeros(self.shape)
            for rx in range(self.shape[0]):
                for ry in range(self.shape[1]):
                    if len (self.d.cal[rx,ry].data)>1:
                        self._inten_tot[rx,ry] = np.sum(self.d.cal[rx,ry].data['intensity'][1:])
        elif seed_picker=='most':
            self._n_data_points = np.zeros(self.shape)
            for rx in range(self.shape[0]):
                for ry in range(self.shape[1]):
                    self._n_data_points[rx,ry] = len(self.d.cal[rx,ry].data['intensity'])
    def set_num_lowq(self,num_lowq):
        self._num_lowq = num_lowq
    def set_num_highi(self,num_highi):
        self._num_highi = num_highi
    def set_a_scale(self,a_scale):
        self._a_scale = a_scale
    def set_store_xtalforms(self,store_xtalforms):
        self.store_xtalforms = store_xtalforms

    # Set data channels
    def get_kpoints(self, p, vp={}, show=True):
        """ p param dict: see get_maxima_2D
            vp param dict: see show
        """
        # find the points
        ans = get_maxima_2D(self.bvm.data, **p)
        self._qx = ans['x']
        self._qy = ans['y']
        self._int = ans['intensity']

        # transform coordinates
        origin = self.d.calibration.get_origin_mean()
        self.origin = np.array([origin[0]*self.upsample, origin[1]*self.upsample])
        self.qx = self._qx - self.origin[0]
        self.qy = self._qy - self.origin[1]
        self.qx *= self.qpixsize
        self.qy *= self.qpixsize

        # store convenience objects
        self._points = np.vstack((self.qx,self.qy)).T
        s = self.d.Qshape
        self.FOV = (s[0]*self.qpixsize*self.upsample, s[1]*self.qpixsize*self.upsample)
        self.Lmax = np.hypot(self.FOV[0],self.FOV[1])/2
        fov = (self.FOV[0]*1.05, self.FOV[1]*1.05)
        self._FOV_centered = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2)

        # get voronoi tesselation
        self._voronoi_points = np.vstack((self._qx,self._qy)).T
        self._voronoi = Voronoi(self._voronoi_points)
        self._voronoi_vertices = get_voronoi_vertices(
            self._voronoi, self.d.Qshape[0], self.d.Qshape[1])

        # show
        print(f"Identified {len(ans)} points")
        if show:
            show_points(
                self.bvm,
                x=self._qx,
                y=self._qy,
                open_circles=True,
                **vp
            )
            self.show_voronoi(
                c='lightcyan',
                lw=0.5,
                vp=vp,
            )

        # Set up masks libraries
        self._masks_library_1D = [None for idx in range(self.N)]
        self._masks_library_2D = [[None for idx in range(self.N)] for jdx in range(self.N)]

    # visualization methods
    def show_labels(self,
        cmap='inferno',
        c_amorph='cornflowerblue',
        vp={'vmin':0,'vmax':4},
        returnfig=False,
        ):
        fig,ax = show(self.labels,mask=~self.empty,
            mask_color=c_amorph,cmap=cmap,returnfig=True, **vp)
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_xtalform(self,
        idx,
        cmap='inferno',
        c_amorph='cornflowerblue',
        c_seed='springgreen',
        marker='x',
        vp={'vmin':0,'vmax':2},
        returnfig=False,
        ):
        coord = self.seeds[idx]
        ar = self.crystalforms[idx]
        print(f'Showing form seeded at {coord}')
        fig,ax = show(ar,mask=~self.empty,
            mask_color=c_amorph,cmap=cmap,returnfig=True, **vp)
        ax.scatter(coord[1],coord[0],color=c_seed,marker=marker)
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_voronoi(self,c='w',lw=1,vp={},returnfig=False):
        # Show
        fig,ax = show_points(
            self.bvm,
            x=self._qx,
            y=self._qy,
            open_circles=True,
            returnfig=True,
            **vp
        )
        for region in range(len(self._voronoi_vertices)):
            vertices_curr = self._voronoi_vertices[region]
            if vertices_curr is not None:
                for i in range(len(vertices_curr)):
                    x0,y0 = vertices_curr[i,:]
                    x1,y1 = vertices_curr[(i+1)%len(vertices_curr),:]
                    ax.plot((y0,y1),(x0,x1),c,lw=lw)
        ax.set_xlim([0,self.bvm.data.shape[1]])
        ax.set_ylim([0,self.bvm.data.shape[0]])
        plt.gca().invert_yaxis()
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_voronoi_mask(self,mask,mask_alpha=0.4,mask_color='y',
        c='lightcyan',lw=0.5,vp={},returnfig=False):
        """ mask is a list of integer (voronoi regions)
        """
        patches = []
        fig,ax = self.show_voronoi(c=c,lw=lw,vp=vp,returnfig=True)
        for idx in mask:
            vertices_curr = self._voronoi_vertices[idx]
            if vertices_curr is not None:
                vert = np.roll(vertices_curr,-1,1)
                patches.append(Polygon(vert))
        p = PatchCollection(patches,alpha=mask_alpha,color=mask_color)
        ax.add_collection(p)
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_data_mask_compare(self,coord,mask_alpha=0.4,mask_color='y',
        c='lightcyan',lw=0.5,marker='x',markercolor='blue',markersize=100,
        vectors=False,vect_cmap='cool',vect_width=0.5,vect_headsize=6,vectp={},
        vp={},returnfig=False):
        """ show the data points, mask, voronoi, bvm overlaid
        """
        rx,ry = coord
        mask = self.state_masks[rx][ry]
        fig,ax = self.show_voronoi_mask(
            mask = mask,
            c = c,
            lw = lw,
            vp = vp,
            mask_alpha=0.4,
            returnfig=True
        )
        qpixsize = self.d.calibration.get_Q_pixel_size()
        origin = self.d.calibration.get_origin_mean()
        d = self.d.cal[rx,ry].data
        x,y = d['qx'],d['qy']
        x,y = self._transform_cal_to_pix(x,y)
        ax.scatter(y,x,color=markercolor,s=markersize,marker=marker)
        # if vectors were requested, add them
        if vectors:
            # set up vectors
            crystals = self.state_crystals[coord[0]][coord[1]]
            crystals_vectors = []
            for xtal in crystals:
                if len(xtal)==1:
                    i = xtal[0]
                    x,y = self.qx[i],self.qy[i]
                    x,y = self._transform_cal_to_pix(x,y)
                    crystals_vectors.append(((x,y),))
                elif len(xtal)==2:
                    i,j = xtal[0],xtal[1]
                    x1,y1 = self.qx[i],self.qy[i]
                    x1,y1 = self._transform_cal_to_pix(x1,y1)
                    x2,y2 = self.qx[j],self.qy[j]
                    x2,y2 = self._transform_cal_to_pix(x2,y2)
                    crystals_vectors.append(((x1,y1),(x2,y2)))
            # set up colors
            l = len(crystals_vectors)
            cm = plt.get_cmap(vect_cmap)
            colors = [cm(n/l) for n in range(l)]
            # plot vectors
            origin = self.d.calibration.get_origin_mean()
            origin=tuple([x*self.upsample for x in origin])
            for xtalvs,color in zip(crystals_vectors,colors):
                for v in xtalvs:
                    add_vector(ax,d=dict({
                        'x0':origin[0],
                        'y0':origin[1],
                        'vx':v[0]-origin[0],
                        'vy':v[1]-origin[1],
                        'color':color,
                        'width':vect_width,
                        'head_width':vect_headsize,
                    }, **vectp))
            print(f"crystal channels: {crystals}")
        # return
        if returnfig:
            return fig,ax
        else:
            plt.show()


    ####### core algorithm methods #######

    def run(self):
        if not self._labelled_empties:
            self.label_empty_pixels()
            self.clean_empty_pixels()
        print('entering seed loop...')
        go_forth = True
        while go_forth:
            go_forth = self._seedloop()
        self._endloop()

    def _seedloop(self):
        # pick a seed...
        l0 = np.sum(self.labels==0)
        l4 = np.sum(self.labels==4)
        # ...from an unassessed pixel
        if l0>0:
            seed = self._new_seed(seed_picker=self.seed_picker)
            self._reset_path_vars()
            #print(f'setting out at seed {seed}')
            self._new_path(seed)
            return True
        # ...from a seeded pixel
        elif l4>0:
            print('seeded path!')
            print('need to make this still - ending...')
            return False # TODO
            seed = self._pick_seeded_seed(seed_picker=self.seed_picker)
            self._reset_path_vars()
            self._seeded_path(seed)
            return True
        # are we done?
        else:
            if not(np.all(self.labels==2)):
                warnings.warn("Warning: finished but not all pixels have been labelled 'complete'")
            return False

    def _new_seed(self,seed_picker='max'):
        pos = np.where(self.labels==0)
        n_empty = len(pos[0])
        assert(n_empty>0), "no 0-labelled pixels found when and empty pixel to seed was requested"
        assert(seed_picker in ['max','most','random','front','back',]), f"Unknown value for seed picker {seed_picker}!"
        if seed_picker == 'random':
            n = np.random.randint(0,n_empty)
        elif seed_picker == 'max':
            n = np.argmax(self._inten_tot[pos])
        elif seed_picker == 'most':
            n = np.argmax(self._n_data_points[pos])
        elif seed_picker == 'back':
            n = -1
        elif seed_picker == 'front':
            n = 0
        else:
            raise Exception(f"Unknown value for seed picker {seed_picker}")
        seed = pos[0][n],pos[1][n]
        return seed

    def _new_path(self,seed):
        """ Setup, then goto _new_path_loop
        """
        # setup
        x,y,inten = self._data_curr = self._get_data(seed)
        self._data_channels_curr = self._get_data_channels(x,y)
        self._b_opts_curr = []
        self._crystals_curr = []
        self._crystals_n_opts = 0
        self._best_score_curr = -1e8
        self._score_cap_curr = 1
        self._mask_base_curr = [0]  # start with center beam masked
        #if self.thresh > 0:
        #    self._mask_base_curr = self._mask_union([self._mask_base_curr,self._get_inten_mask(inten)])
        self._mask_curr = self._mask_base_curr
        # loop, find crystals
        loop_path = True
        while loop_path:
            loop_path = self._new_path_loop(seed)
        return

    def _new_path_loop(self,seed):
        """ Find a crystal set and mask, then _put_on_shoes_and_coat
        """
        x,y,inten = self._data_curr
        # prepare boolean mask
        # length matches datapoints and True indicates unmasked
        m = np.ones(len(x),dtype=bool)
        for idx in range(len(x)):
            if self._data_channels_curr[idx] in self._mask_curr:
                m[idx] = 0
            if self._data_channels_curr[idx] in self._b_opts_curr:
                m[idx] = 0
        # check for data, label and exit if there isn't any
        if len(x[m]) < self.min:
            self.labels[seed[0],seed[1]] = 2
            self.empty[seed[0],seed[1]] = 2
            return False
        # find crystals
        # first get basis vector options & new crystal options
        # then compute permutation scores and update vars
        self._crystals_n_opts += 1
        self._get_b_next_opts(m)
        crystal_opts, mask_opts = self._get_crystal_opts_curr()
        self._score_crystal_options_and_update(crystal_opts,mask_opts)
        # are we done?
        if self._best_score_curr < -1e6 * self._score_cap_curr:
            # ...no?  iterate
            return True
        else:
            # ...yes? finish
            for xtal in self._crystals_curr:
                self.crystals.append(xtal)
                self.crystal_seeds.append(seed)
            self.state_crystals[seed[0]][seed[1]] = self._crystals_curr
            self.state_masks[seed[0]][seed[1]] = self._mask_curr
            self._scores[seed[0],seed[1]] = self._best_score_curr
            self.labels[seed[0],seed[1]] = 1
            # done
            self._put_on_shoes_and_coat(seed)
            return False

    def _put_on_shoes_and_coat(self,seed):
        """ get ready to walk. then, go walking
        """
        self._reset_path_vars()
        self._coords.append(seed)
        self._dirs.append(0)
        go_walking = True
        while go_walking:
            go_walking = self._walk()
        # and back again
        go_home = True
        while go_home:
            go_home = self._backtrack_and_spawn()
            if go_home:
                go_walking = True
                while go_walking:
                    go_walking = self._walk()

    def _walk(self):
        """ look at the label of the pixel ahead.
        is it 0? new pixel - check for mask change, then walk or turn
        is it 1,2,3 or an edge? turn
        is it 3? merge seeds, then walk
        """
        d = self._dirs[self._pos]
        c,l = self._look_ahead()
        # catch for lone pixels
        if len(self._coords)==1 and d==3:
            self._single_pixel_shake()
            return False
        # backtracking
        if len(self._coords)>1:
            if np.array_equal(np.array(c),np.array(self._coords[-2])):
                return False
        # unavalable - turn right
        if l in (-1,1,2,3):
            self._dirs[self._pos] = self._right(d)
            return True
        # open - onward!
        elif l in (0,4):
            self._new_pixel(c)
            return True
        else:
            raise Exception(f'encountered unexpected label {l}')

    def _seeded_path(self,seed):
        """ setup variables, gather neighbor xtals & masks, then enter the path loop
        """
        # setup - set data
        x,y,inten = self._data_curr = self._get_data(seed)
        self._data_channels_curr = self._get_data_channels(x,y)
        # reset vars
        self._crystals_curr = []
        self._best_score_curr = -1e8
        self._crystal_gen_iters = 0
        self._score_cap_curr = 1
        self._mask_base_curr = [0]  # start with center beam masked
        if self.thresh > 0:
            self._mask_base_curr = self._mask_union([self._mask_base_curr,self._get_inten_mask(inten)])
        self._mask_curr = self._mask_base_curr
        # gather xtals and masks from neighbors
        xtals = []
        masks = []
        coords = [
            (seed[0]-1,seed[1]),
            (seed[0],seed[1]-1),
            (seed[0]+1,seed[1]),
            (seed[0],seed[1]+1)]
        s = self.shape
        for idx in range(3,-1,-1):
            c = coords[idx]
            if c[0]<0 or c[0]>=s[0] or c[1]<0 or c[1]>=s[1]:
                coords.pop(idx)
        for c in coords:
            if self.labels[c[0],c[1]] == 2:
                xtals.append(self.state_crystals[c[0]][c[1]])
                masks.append(self.state_masks[c[0]][c[1]])
        # confirm we found some xtals
        assert(len(xtals)>0), "this shouldn't be able to happen..."
        assert(len(xtals)>0), "you can initiate a _new_path_loop instead, but do check labels."
        # prepare mask
        m = np.ones(len(x),dtype=bool)
        for idx in range(len(x)):
            if self._data_channels_curr[idx] in self._mask_curr:
                m[idx] = 0
        # check for data, and if there's none label & exit
        if len(x[m]) < self.min:
            self.labels[seed[0],seed[1]] = 2
            self.empty[seed[0],seed[1]] = 2
            return False
        # setup crystals & masks from adjacent pixels
        xtals_compare = []
        masks_compare = []
        for _xtals,_masks in zip(xtals,masks):
            for xtal,mask in zip(_xtals,_masks):
                if not self._xtal_in_xtals(xtal,xtals_compare):
                    xtals_compare.append(xtal)
                    masks_compare.append(mask)
        # compare and update
        self._score_crystal_options_and_update(xtals_compare,masks_compare)
        # for scores above -1e6,
        # check for excess crystals,
        # then return
        if self._best_score_curr > -1e6 * self._score_cap_curr:
            # remove loop
            can_remove, idx_rm = self._can_remove_crystal(self._data_channels_curr)
            while can_remove:
                self._crystals_curr.pop(idx,rm)
                can_remove, idx_rm = self._can_remove_crystal(self._data_channels_curr)
            # finish
            for xtal in self._crystals_curr:
                if not self._xtal_in_xtals(xtal, self.crystals):
                    self.crystals.append(xtal)
            self.state_crystals[seed[0]][seed[1]] = self._crystals_curr
            self.state_masks[seed[0]][seed[1]] = self._mask_curr
            self._scores[seed[0],seed[1]] = self._best_score_curr
            self.labels[seed[0],seed[1]] = 1
            # done
            self._put_on_shoes_and_coat(seed)
            return False
        # for scores below -1e6, i.e. missing data points,
        # is the score within tolerances?
        # if so try a rm, then return
        if self._best_score_curr < -1e6 * self._score_cap_curr:
            all_clear = True
            channels_unmasked = list(set(self._data_channels_curr)-set(self._mask_curr))
            channels_unmasked_bool = np.array([c in channels_unmasked for c in self._data_channels_curr])
            for idx in np.nonzero(channels_unmasked_bool)[0]:
                # intensity thresh
                if inten[idx] < self.thresh:
                    pass
                # distance tolerance
                else:
                    _x,_y = x[idx],y[idx]
                    _c = channels[idx]
                    dist_unmasked = np.hypot(_x-self.qx[_c],_y-self.qy[_c])
                    dist_masked = 1e8
                    for ch in self._mask_curr:
                        _d = np.hypot(_x-self.qx[ch],_y-self.qy[ch])
                        if _d<dist_masked:
                            dist_masked = _d
                    dist_frac_tol = dist_masked/dist_unmasked
                    if dist_frac_tol > self.distance_frac_tolerance:
                        all_clear = False
            # number threshold
            if not all_clear:
                _frac = len(channels_unmasked)/len(channels)
                if _frac < self.number_frac_tolerance:
                    all_clear = True
            if all_clear:
                # remove loop
                can_remove, idx_rm = self._can_remove_crystal(self._data_channels_curr)
                while can_remove:
                    self._crystals_curr.pop(idx,rm)
                    can_remove, idx_rm = self._can_remove_crystal(self._data_channels_curr)
                # assign labels
                for xtal in self._crystals_curr:
                    if not self._xtal_in_xtals(xtal, self.crystals):
                        self.crystals.append(xtal)
                self.state_crystals[seed[0]][seed[1]] = self._crystals_curr
                self.state_masks[seed[0]][seed[1]] = self._mask_curr
                self._scores[seed[0],seed[1]] = self._best_score_curr
                self.labels[seed[0],seed[1]] = 1
                # go walking
                self._put_on_shoes_and_coat(seed)
                # return
                return False
        # for scores below -1e6 beyond tolerances,
        # gen a new solution, then return
        # reset solution vars
        self._crystals_curr = []
        self._best_score_curr = -1e8
        self._crystal_gen_iters = 0
        self._score_cap_curr = 1
        self._mask_curr = self._mask_base_curr
        loop = True
        while loop:
            loop = self._seeded_path_internal_loop(seed,x,y,inten,xtals,masks)
        return

    def _seeded_path_internal_loop(self,seed,x,y,inten,xtals,masks):
        """ gen new solution if other options fail, then walk
        """
        # get basis vector options & new crystal options
        b_next_opts = self._get_b_next_opts(self._mask_curr) # TODO - mask wrong, needs boolean
        crystals_opts,mask_opts = self._get_crystal_opts_add(b_next_opts)
        # compute scores and update
        self._score_crystal_options_and_update(crystals_opts,mask_opts)
        # are we done?
        if self._best_score_curr > -1e6 * self._score_cap_curr:
            # finish
            for xtal in self._crystals_curr:
                if not self._xtal_in_xtals(xtal, self.crystals):
                    self.crystals.append(xtal)
            self.state_crystals[seed[0]][seed[1]] = self._crystals_curr
            self.state_masks[seed[0]][seed[1]] = self._mask_curr
            self._scores[seed[0],seed[1]] = self._best_score_curr
            self.labels[seed[0],seed[1]] = 1
            # done
            self._put_on_shoes_and_coat(seed)
            return False
        else:
            # iterate
            self._crystal_gen_iters += 1
            if self._crystal_gen_iters%10 != 0:
                return True
            elif self._crystal_gen_iters != 100:
                # increase the score cap
                self._score_cap_curr += 1
                print(f'increased cap to {self._score_cap_curr}')
                self._score_caps[seed[0],seed[1]] += 1
                #  reset the vectors and xtal options
                self._crystals_curr = []
                self._mask_curr = self._mask_base_curr
                self._best_score_curr = -1e8
                return True
            else:
                print('she cant take much more o this, capn')
                self._anomoly = True
                self._anom_point = seed
                raise Exception(f'an unexpected error has occured; the crystal gen algo needs attention')
                sys.exit()
                return False

    def _pick_seeded_seed(self,seed_picker='max'):
        pos = np.where(self.labels==4)
        n_seeds = len(pos[0])
        assert(n_seeds>0), "no 3-labelled pixels found when a seeded pixel was requested"
        assert(seed_picker in ['max','random','front','back'])
        if seed_picker == 'random':
            n = np.random.randint(0,n_seeds)
        elif seed_picker == 'max':
            maxi = 0
            for idx in range(len(pos[0])):
                val = np.sum(self.d.cal[pos[0][idx],pos[1][idx]].data['intensity'])
                if val>maxi:
                    maxi = val
                    n = idx
        elif seed_picker == 'back':
            n = -1
        elif seed_picker == 'front':
            n = 0
        else:
            raise Exception(f"Unknown value for seed picker {seed_picker}")
        seed = pos[0][n],pos[1][n]
        return seed

    def _new_pixel(self,coord):
        """
        """
        # get data
        x,y,inten = self._data_curr = self._get_data(coord)
        channels = self._get_data_channels(x,y)
        # get the cost
        cost = self._get_cost(channels,self._crystals_curr,self._mask_curr)
        # for scores below -1e6, i.e. missing data points,
        # are we within tolerances? if so, label & return
        if cost < -1e6 * self._score_cap_curr:
            all_clear = True
            channels_unmasked = list(set(channels)-set(self._mask_curr))
            channels_unmasked_bool = np.array([c in channels_unmasked for c in channels])
            for idx in np.nonzero(channels_unmasked_bool)[0]:
                # intensity thresh
                if inten[idx] < self.thresh:
                    pass
                # distance tolerance
                else:
                    _x,_y = x[idx],y[idx]
                    _c = channels[idx]
                    dist_unmasked = np.hypot(_x-self.qx[_c],_y-self.qy[_c])
                    dist_masked = 1e8
                    for ch in self._mask_curr:
                        _d = np.hypot(_x-self.qx[ch],_y-self.qy[ch])
                        if _d<dist_masked:
                            dist_masked = _d
                    dist_frac_tol = dist_masked/dist_unmasked
                    if dist_frac_tol > self.distance_frac_tolerance:
                        all_clear = False
            # number threshold
            if not all_clear:
                _frac = len(channels_unmasked)/len(channels)
                if _frac < self.number_frac_tolerance:
                    all_clear = True
            # tag, turn, & return
            if not all_clear:
                self.labels[coord[0],coord[1]] = 3
                self._dirs[self._pos] = self._right(self._dirs[self._pos])
                return True
        # if no data points are missing, can we remove crystals?
        # if so, label 3, turn walk & return
        if self._can_remove_crystal(channels)[0]:
            self.labels[coord[0],coord[1]] = 3
            self._dirs[self._pos] = self._right(self._dirs[self._pos])
            return True
        else:
            # if no data is missing and no crystals can be removed,
            # label, then continue  walking - step forward, turn, walk
            self.state_crystals[coord[0]][coord[1]] = self._crystals_curr
            self.state_masks[coord[0]][coord[1]] = self._mask_curr
            self._scores[coord[0]][coord[1]] = self._best_score_curr
            self._score_caps[coord[0],coord[1]] = self._score_cap_curr
            self.labels[coord[0],coord[1]] = 1
            # update path
            self._coords.append(coord)
            self._pos += 1
            self._dirs.append(self._left(self._dirs[self._pos-1]))
            return True
        pass

    def _backtrack_and_spawn(self):
        # are we done?
        if self._pos == 0:
            self._clean_path(self._coords[0])
            return False
        else:
            # no? orient, step backwards, and remove path end point
            p = self._pos
            c_curr = self._coords[p]
            c_prev = self._coords[p-1]
            d = self._get_dir_towards(c_curr,c_prev)
            self._coords.pop(p)
            self._dirs.pop(p)
            self._pos -= 1
            # turn left and walk
            self._dirs[p-1] = self._left(d)
            return True

    def _endloop(self):
        """ cleans up
        """
        # TODO
        # wrap in an access class?
        print('holy cow, its the endloop whaaaaaaaaaaaaaaaat')
        pass


    ####### scoring utilities #######

    def _score_crystal_options_and_update(self,crystals_opts,mask_opts):
        """ compare the scores of all the current options and
        update with the current best option
        """
        for xtals_opt,mask_opt in zip(crystals_opts,mask_opts):
            cost = self._get_cost(self._data_channels_curr,xtals_opt,mask_opt)
            if cost > self._best_score_curr:
                self._crystals_curr = xtals_opt
                self._mask_curr = mask_opt
                self._best_score_curr = cost

    # crystal generation permutations
    def _get_b_next_opts(self,m):
        """ finds the next basis vector options and adds them to the running list
        of options self._b_opts_curr.  m is boolean with len(data)
        """
        # setup
        x,y,inten = self._data_curr
        m_inds = np.nonzero(m)[0]
        n = np.sum(m)
        N = self._crystals_n_opts
        # for new set, populate with num_lowq + num_highi vectors
        if len(self._b_opts_curr)==0:
            num_lowq = self._num_lowq if self._num_lowq<n else n-1
            num_highi = self._num_highi if self._num_highi<n else n-1
            # get low q options
            q = np.hypot(x[m],y[m])
            qsort_inds = np.argpartition(q,num_lowq)  # lowest to highest q
            _opts = list(qsort_inds[:num_lowq])
            # get high inten options
            isort_inds = np.argpartition(inten[m],-num_highi)
            _opts += list(isort_inds[-num_highi:])
            # rm redundancies
            opts = []
            for opt in _opts:
                if opt not in opts:
                    opts.append(opt)
        # for existing set, add 1 new low q and 1 new high i
        else:
            # low q
            q = np.hypot(x[m],y[m])
            opts = [np.argmin(q)]
            # high inten
            ind = np.argmax(inten[m])
            if ind not in opts:
                opts += [ind]
        # transform to voronoi channel indices
        opts_indices = []
        for opt in opts:
            idx = m_inds[opt]
            opts_indices.append(self._get_channel((x[idx],y[idx])))
        # merge with current best of b options
        # new set
        if len(self._b_opts_curr)==0:
            self._b_opts_curr += opts_indices
        # existing set
        else:
            for opt in opts_indices:
                if opt not in self._b_opts_curr:
                    self._b_opts_curr.append(opt)
        # return
        return self._b_opts_curr

    def _get_crystal_opts_curr(self):
        """ get all crystal options of the current N crystals
        given the current b-vectors
        """
        b_opts = self._b_opts_curr
        crystals_opts = []
        masks_opts = []
        # ensure we have enough b options
        N = self._crystals_n_opts
        assert(len(b_opts)>=N), "number of b options should not be less than number of crystals!"
        # get b options combinations
        b_combs = list(combinations(b_opts,N))
        # get on-/off- axis set possibilities
        n_on_max = N//2
        n_onoff_sets = n_on_max+1
        onoff_sets = []
        for idx in range(n_onoff_sets):
            onoff_sets.append( tuple([2]*idx + [1]*(N-2*idx)) )
        # combine b option combs with on/off-axis combs to get all combinations
        for onoff in onoff_sets:
            n_onoff_curr = len(onoff)
            onoff_permutes = set(permutations(onoff,n_onoff_curr))
            for onoff_permute_option in onoff_permutes:
                for b_comb in b_combs:
                    xtal_opt_curr = []
                    idx = 0
                    for val in onoff_permute_option:
                        if val==1:
                            xtal_opt_curr += [(b_comb[idx],)]
                            idx += 1
                        else:
                            xtal_opt_curr += [(b_comb[idx],b_comb[idx+1])]
                            idx += 2
                    crystals_opts.append( xtal_opt_curr )
                    # get mask and append
                    masks_xtals_curr = [self._get_mask(xtal) for xtal in xtal_opt_curr]
                    mask_curr = self._mask_union([self._mask_base_curr]+masks_xtals_curr)
                    masks_opts.append(mask_curr)
        # return
        return crystals_opts, masks_opts


    ####### other utility methods #######

    # data
    def _get_data(self,seed):
        data = self.d.cal[seed[0],seed[1]]
        return data.data['qx'],data.data['qy'],data.data['intensity']

    def _get_data_channels(self,x,y):
        """ takes x,y positions and returns their voronoi channel indices
        """
        channels = []
        l = len(x)
        for idx in range(l):
            channels.append(self._get_channel((x[idx],y[idx])))
        return channels

    def _get_channel(self,q):
        """ accepts a tuple q=(qx,qy) and returns the index of
        its Voronoi region
        """
        return np.argmin(np.hypot(self.qx-q[0],self.qy-q[1]))

    # cost
    def _get_cost(self, channels, xtals, mask):
        """ The cost is
                f = -1e6a-1e3b-c
        where
            a = # unmasked data points (v1)
            a = sum of unmasked data intensities (v2)
            b = # basis vectors
            c = # on-axis crystals
        """
        a = 0
        for idx,c in enumerate(channels):
            if c not in mask:
                a += self._data_curr[2][idx]
        #a = len(set(channels)-set(mask))
        b = 0
        c = 0
        for xtal in xtals:
            l = len(xtal)
            b += l
            if l>1:
                c += 1
        return -1e6*a*self._a_scale-1e3*b-c

    # empty pixels
    def label_empty_pixels(self):
        self._labelled_empties = True
        s = self.d.shape
        for rx in range(s[0]):
            for ry in range(s[1]):
                l = len(self.d.cal[rx,ry].data)
                i = np.sum(self.d.cal[rx,ry].data['intensity'][1:])
                if l < self.min_empty:
                    self.labels[rx,ry] = 2
                    self.empty[rx,ry] = 1
                elif i < self.thresh_empty:
                    self.labels[rx,ry] = 2
                    self.empty[rx,ry] = 1

    def clean_empty_pixels(self,reverse=False,iteropen=1,iterclose=1):
        """ Performs a binary opening then closing. `reverse` flipse the order,
            and iter* control iterations
        """
        if reverse:
            x = binary_closing(binary_opening(
                self.labels,iterations=iteropen),iterations=iterclose)
        else:
            x = binary_opening(binary_closing(
                self.labels,iterations=iterclose),iterations=iteropen)
        self.empty = x
        self.labels[x] = 2
        self.labels[np.logical_not(x)] = 0

    # boolean checks
    def _can_remove_crystal(self,channels):
        """ can we?
        """
        for idx in range(len(self._crystals_curr)):
            xtals = self._crystals_curr.copy()
            xtals.pop(idx)
            masks = [self._get_mask(xtal) for xtal in xtals]
            mask = self._mask_union([self._mask_base_curr]+masks)
            cost = self._get_cost(channels,xtals,mask)
            if cost > -1e6 * self._score_cap_curr:
                return True, idx
        return False, -1

    def _xtal_in_xtals(self,xtal,xtals):
        if len(xtal)==1:
            if xtal in xtals:
                return True
        else:
            if xtal in xtals or (xtal[1],xtal[0]) in xtals:
                return True
        return False

    def _mask_is_already_present(self, mask, mask_compare):
        if len(set(mask)-set(mask_compare))>0:
            return False
        else:
            return True

    # masks
    def _get_mask(self,crystal):
        """ crystal should be a tuple of 1 or 2 ints (off/on-axis) indicating basis vectors
        as indices in the voronoi.points
        """
        l = len(crystal)
        if l == 1:
            return self._get_mask_1D(crystal[0])
        elif l == 2:
            return self._get_mask_2D(crystal[0],crystal[1])
        else:
            raise Exception(f"Length of a crystal should be 1 or 2, not {l}")

    def _get_mask_1D(self, b):
        """ b is an integer indexing a voronoi point to serve as the basis vector
        """
        val = self._masks_library_1D[b]
        if val is not None:
            return val
        else:
            q = self._points[b,:]
            l = np.hypot(q[0],q[1])
            # tile in 1D
            ntiles = int(np.ceil(self.Lmax/l))+1
            points = [(q[0]*n,q[1]*n) for n in range(1,ntiles)]
            points += [(-q[0]*n,-q[1]*n) for n in range(1,ntiles)]
            # remove points beyond FOV
            rm = []
            for idx,point in enumerate(points):
                f = self._FOV_centered
                if point[0]<f[0] or point[0]>=f[1] or point[1]<f[2] or point[1]>=f[3]:
                    rm.append(idx)
            while len(rm)>0:
                idx = rm.pop(-1)
                points.pop(idx)
            # get mask
            mask = []
            for p in points:
                mask.append(self._get_channel(p))
            # return
            self._masks_library_1D[b] = mask
            return mask

    def _get_mask_2D(self, b1, b2):
        """ b1 and b2 are integers indexing a voronoi point to serve as the basis vector
        """
        val = self._masks_library_2D[b1][b2]
        if val is not None:
            return val
        else:
            q1 = self._points[b1,:]
            q2 = self._points[b2,:]
            l1 = np.hypot(q1[0],q1[1])
            l2 = np.hypot(q2[0],q2[1])
            # tile in 2D
            ntiles1 = int(np.ceil(self.Lmax/l1))+1
            ntiles2 = int(np.ceil(self.Lmax/l2))+1
            points = []
            for n in range(-ntiles1-1,ntiles1):
                for m in range(-ntiles2-1,ntiles2):
                    if not(n==0 and m==0):
                        points.append((q1[0]*n+q2[0]*m, q1[1]*n+q2[1]*m))
            # remove points beyond FOV
            rm = []
            for idx,point in enumerate(points):
                f = self._FOV_centered
                if point[0]<f[0] or point[0]>=f[1] or point[1]<f[2] or point[1]>=f[3]:
                    rm.append(idx)
            while len(rm)>0:
                idx = rm.pop(-1)
                points.pop(idx)
            # get mask
            mask = []
            for p in points:
                mask.append(self._get_channel(p))
            # return
            self._masks_library_2D[b1][b2] = self._masks_library_2D[b2][b1] = mask
            return mask

    def _get_inten_mask(self, inten):
        return np.nonzero(inten>self.thresh)[0]

    def _mask_union(self, masks):
        """ masks should be a list of masks; each mask is itself a list of integers
        """
        m = set([])
        for mask in masks:
            m = m | set(mask)
        return list(m)

    # orienteering
    # comment: let (0,1,2,3) be down, left, up, right
    # in UL origin vertical x RHC system
    def _left(self,n):
        return (n-1)%4
    def _right(self,n):
        return (n+1)%4

    def _get_dir_towards(self,coord1,coord2):
        """ returns the direction from coord1 to coord2
        """
        d0 = coord2[0]-coord1[0]
        d1 = coord2[1]-coord1[1]
        assert(set([0,1]) == set([abs(d0),abs(d1)])), f"invalid coord pair on path {coord1} {coord2}"
        match (d0,d1):
            case (1,0):
                return 0
            case (0,-1):
                return 1
            case (-1,0):
                return 2
            case (0,1):
                return 3

    def _look_ahead(self):
        """ return the position and label ahead
        """
        c = np.array(self._coords[self._pos])
        d = self._dirs[self._pos]
        match d:
            # down
            case 0:
                c[0] += 1
            # left
            case 1:
                c[1] -= 1
            # up
            case 2:
                c[0] -= 1
            # right
            case 3:
                c[1] += 1
            case _:
                raise Exception(f'invalid direction {d}!')
        # frame edge
        s = self.shape
        if c[0]<0 or c[1]<0 or c[0]>=s[0] or c[1]>=s[1]:
            return -1,-1
        # otherwise, return
        else:
            return c,self.labels[c[0],c[1]]

    # miscellaneous
    def _clean_path(self,seed):
        self.seeds.append(seed)
        lab1 = self.labels==1
        lab3 = self.labels==3
        if self.store_xtalforms:
            ar = np.zeros(self.shape,dtype=int)
            ar[lab1]=1
            ar[lab3]=2
            self.crystalforms.append(ar)
        self.labels[lab1] = 2
        self.labels[lab3] = 4
        self._reset_path_vars()
        pass

    def _single_pixel_shake(self):
        #print('\/\** \/ **/\/ ***do the single*pixel shake*** \/\** \/**\/ **/\/')
        pass

    def _transform_cal_to_pix(self,x,y):
        """ from calibrated to upsampled Qspace pixel coordinates
        """
        qpixsize = self.d.calibration.get_Q_pixel_size()
        origin = self.d.calibration.get_origin_mean()
        x /= self.qpixsize
        y /= self.qpixsize
        x += origin[0]*self.upsample
        y += origin[1]*self.upsample
        return x,y




# code graveyard

            # iterate
            #self._crystal_gen_iters += 1
            #if self._crystal_gen_iters%10 != 0:
            #    return True
            #elif self._crystal_gen_iters != 100:
            #    # increase the score cap
            #    self._score_cap_curr += 1
            #    print(f'increased cap to {self._score_cap_curr}')
            #    self._score_caps[seed[0],seed[1]] += 1
            #    #  reset the vectors and xtal options
            #    self._crystals_curr = []
            #    self._mask_curr = self._mask_base_curr
            #    self._best_score_curr = -1e8
            #    return True
            #else:
            #    print('she cant take much more o this, capn')
            #    self._anomoly = True
            #    self._anom_point = seed
            #    raise Exception(f'an unexpected error has occured; the crystal gen algo needs attention')
            #    sys.exit()
            #    return False


        #b_next_opts = self._get_b_next_opts(m)
        #crystals_opts,mask_opts = self._get_crystal_opts_add(b_next_opts)
        #self._score_crystal_options_and_update(crystals_opts,mask_opts)


        # iterate over combinations
        #l = len(crystals_opts)
        #for L in range(1,l+1):
        #    for comb in combinations(range(l),r=L):
        #        xtals = [crystals_opts[x] for x in comb]
        #        masks = [mask_opts[x] for x in comb]
        #        mask = self._mask_union([self._mask_base_curr]+masks)
        #        cost = self._get_cost(self._data_channels_curr,xtals,mask)
        #        if cost > self._best_score_curr:
        #            self._crystals_curr = xtals
        #            self._mask_curr = mask
        #            self._best_score_curr = cost


#     def _get_crystal_opts_add(self, b_next_opts):
#         crystals_opts = []
#         mask_opts = []
#         # new
#         if len(self._crystals_curr)==0:
#             for bv in b_next_opts:
#                 m = self._get_mask((bv,))
#                 masks_so_far = self._mask_union([self._mask_curr]+mask_opts)
#                 if not self._mask_is_already_present(m,masks_so_far):
#                     crystals_opts.append((bv,))
#                     mask_opts.append(m)
#         else:
#             # try adding new 1D crystals
#             for bv in b_next_opts:
#                 m = self._get_mask((bv,))
#                 masks_so_far = self._mask_union([self._mask_curr]+mask_opts)
#                 if not self._mask_is_already_present(m,masks_so_far):
#                     crystals_opts.append((bv,))
#                     mask_opts.append(m)
#             # try making an existing 1D crystal a 2D crystal
#             for xtal in self._crystals_curr:
#                 if len(xtal)==1:
#                     for bv in b_next_opts:
#                         m = self._get_mask((xtal[0],bv))
#                         masks_so_far = self._mask_union([self._mask_curr]+mask_opts)
#                         if not self._mask_is_already_present(m,masks_so_far):
#                             crystals_opts.append((xtal[0],bv))
#                             mask_opts.append(m)
#         # return
#         return crystals_opts,mask_opts


