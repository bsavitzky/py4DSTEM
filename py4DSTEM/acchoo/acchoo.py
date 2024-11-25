import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi
from scipy.ndimage import binary_opening, binary_closing
from py4DSTEM.utils import get_maxima_2D, get_voronoi_vertices
from py4DSTEM.visualize import show, show_points

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
    """
    def __init__(
        self,
        disks,
        upsample=1,
        min_points=3,
        min_inten_empty=0,
        min_inten=0,
        seed_picker='max',
        ):
        """
        Parameters
        ----------
        disks : BraggVectors
        upsample : integer
            upsample factor for the Bragg vector histogram
        min_points : integer
            threshold # data points for 'empty' pixels
        min_inten_empty : float
            threshold intensity for data points to be counted as 'empty' or not
        min_inten : bool
            data inclusion intensity threshold
        seed_picker : str in 'max' or 'random'
            strategy for choosing crystal seed points
        """
        self._setup_disks(disks,upsample)
        self._reset_path_vars()
        self._setup()
        self.set_min_points(min_points)
        self.set_min_inten_empty(min_inten_empty)
        self.set_min_inten(min_inten)
        self.set_seed_picker(seed_picker)

    def _setup_disks(self, disks, upsample=1):
        self.d = disks
        self.upsample = upsample
        self.bvm = disks.histogram(sampling=upsample)

    def _reset_path_vars(self):
        self.pos = 0
        self.dirs = []
        self.coords = []

    def _setup(self):
        s = self.shape
        self.labels = np.zeros(s,dtype=int)
        self.state_crystals = [[[] for y in range(s[1])] for x in range(s[0])]
        self.state_masks = [[[] for y in range(s[1])] for x in range(s[0])]
        self.crystals = []
        self.seed_queue = []
        self.basis_vects = []
        self.labelled_empties = False

    @property
    def N(self):
        return len(self._voronoi.points)

    @property
    def qpixsize(self):
        return self.d.calibration.get_Q_pixel_size()/self.upsample

    @property
    def shape(self):
        return self.d.shape

    def set_min_points(self,min_points):
        self.min = min_points

    def set_min_inten_empty(self,min_inten_empty):
        self.thresh_empty = min_inten_empty

    def set_min_inten(self,min_inten):
        self.thresh = min_inten

    def set_seed_picker(self,seed_picker):
        assert(seed_picker in ['max','random',]), f"Unknown value for seed picker {seed_picker}!"
        self.seed_picker = seed_picker

    def show_labels(self, vp={}, returnfig=False):
        fig,ax = show(self.labels, returnfig=True, **vp)
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def get_kpoints(self, p, vp={}):
        """ p param dict: see get_maxima_2D
            vp param dict: see show
        """
        ans = get_maxima_2D(self.bvm.data, **p)
        self._qx = ans['x']
        self._qy = ans['y']
        self._int = ans['intensity']

        origin = self.d.calibration.get_origin_mean()
        self.origin = np.array([origin[0]*self.upsample, origin[1]*self.upsample])
        self.qx = self._qx - self.origin[0]
        self.qy = self._qy - self.origin[1]
        self.qx *= self.qpixsize
        self.qy *= self.qpixsize

        # convenience objects
        self._points = np.vstack((self.qx,self.qy)).T
        s = self.d.Qshape
        self.FOV = (s[0]*self.qpixsize*self.upsample, s[1]*self.qpixsize*self.upsample)
        self.Lmax = np.hypot(self.FOV[0],self.FOV[1])/2
        fov = (self.FOV[0]*1.05, self.FOV[1]*1.05)
        self._FOV_centered = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2)

        print(f"Identified {len(ans)} points")
        show_points(
            self.bvm,
            x=self._qx,
            y=self._qy,
            open_circles=True,
            **vp
        )

        # Get voronoi tesselation
        self._voronoi_points = np.vstack((self._qx,self._qy)).T
        self._voronoi = Voronoi(self._voronoi_points)
        self._voronoi_vertices = get_voronoi_vertices(
            self._voronoi, self.d.Qshape[0], self.d.Qshape[1])

        # Set up masks libraries
        self._masks_library_1D = [None for idx in range(self.N)]
        self._masks_library_2D = [[None for idx in range(self.N)] for jdx in range(self.N)]

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

    def label_empty_pixels(self):
        self.labelled_empties = True
        s = self.d.shape
        for rx in range(s[0]):
            for ry in range(s[1]):
                l = len(self.d.cal[rx,ry].data)
                i = np.sum(self.d.cal[rx,ry].data['intensity'][1:])
                if l < self.min:
                    self.labels[rx,ry] = 1
                elif i < self.thresh_empty:
                    self.labels[rx,ry] = 1

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
        self.labels[x] = 1
        self.labels[np.logical_not(x)] = 0

    def run(self):
        if not self.labelled_empties:
            self.label_empty_pixels()
        #self._seedloop()
        #self._walk()
        #self._endloop()


    def _seedloop(self):
        # are we done?
        n_seeds = len(self.seed_queue)
        if n_seeds == 0:
            n_empty_pix = np.sum(self.labels==0)
            if n_empty_pix == 0:
                assert(np.all(self.labels==1)), "Seed queue is empty but unlabeled pixels remain"
                return
            else:
               self._new_seed(seed_picker=self.seed_picker)
        # new path
        seed = self.seed_queue.pop(0)
        self._reset_path_vars()
        label = self.labels[seed[0],seed[1]]
        if label == 0:
            self._new_path(seed)
        elif label in (2,3):
            self._seeded_path(seed)
        else:
            raise Exception(f"unexpected label {label} encountered during new path creation")

    def _new_seed(self,seed_picker='max'):
        # add a seed to the seed queue
        pos = np.where(self.labels==0)
        n_empty = len(pos[0])
        assert(n_empty>0), "no 0-labeled pixels found when new seed was requested"
        assert(seed_picker in ['max','random',]), f"Unknown value for seed picker {seed_picker}!"
        if seed_picker == 'random':
            n = np.random.randint(0,n_empty)
        elif seed_picker == 'max':
            n = np.argmax(np.sum(self.disks.cal[pos[0][:],pos[1][:]].data['intensity']))
        else:
            raise Exception(f"Unknown value for seed picker {seed_picker}")
        seed = pos[0][n],pos[1][n]
        self._seed_queue.append(seed)

    def _new_path(self,seed,crystals=[],best_set=[],mask=[0],masks=[],need_inten_mask=True):
        """ Find and return a minimal basis set and mask.
        Initialized with the zero beam masked
        """
        # update
        x,y,inten = self._get_data(seed)
        self._crystals_curr = crystals
        self._best_crystal_set_curr = best_set
        self._best_score_curr
        self._mask_curr = mask
        self._data_channels_curr = self._get_data_channels(x,y)
        if self.thresh > 0:
            if need_inten_mask:
                self._mask_inten_curr = self._get_inten_mask(self.thresh)
                need_inten_mask = False
            self._mask_curr = self._mask_union([self._mask_curr,self._mask_inten_curr])
        # get basis vector and mask options and compare
        b_next_opts = self._get_b_next_opts(x,y,inten,self._mask_curr)
        crystal_opts,mask_opts = self._get_crystal_opts_add(b_next_opts)
        self._score_crystal_options_and_update(crystal_opts,mask_opts)
        # check if we've masked every data point and if so, return
        if self._best_score_curr > -1e6:
            self.state_crystals[seed[0]][seed[1]] = self._best_crystal_set_curr
            self.state_mask[seed[0]][seed[1]] = self._mask_curr
            return



        #self._bs_curr.append(b_next)
        #mask_opts = self._get_mask_permutations(self.bs_curr)
        #scores = self._get_costs(mask_opts)
        #ind = np.argmin(scores)
        #if scores[ind] > -1000000:
        #   # End - return ans
        #   pass
        #else:
        #   self._new_path(seed,bs=bs)
        pass

    def _get_data(self,seed):
        data = self.d.cal[seed[0],seed[1]]
        return data['qx'],data['qy'],data['intensity']

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

    def _get_inten_mask(self, inten):
        return np.nonzero(inten>self.thresh)[0]

    def _mask_union(self, masks):
        """ masks should be a list of masks; each mask is itself a list of integers
        """
        m = set([])
        for mask in masks:
            m = m | set(mask)
        return list(m)

    def _get_b_next_opts(self,x,y,inten):
        """ returns the next basis vectors to choose from
        """
        # prepare mask
        m = np.ones(len(x),dtype=bool)
        m[self._mask_curr] = 0
        m_inds = np.nonzero(m)[0]
        # get options
        q = np.hypot(x[m],y[m])
        opt1 = np.argmin(q)
        q[opt1] = 1e6
        opt2 = np.argmin(q)
        i = inten[m]
        opt3 = np.argmax(i)
        opts = []
        for opt in (opt1,opt2,opt3):
            if opt not in opts:
                opts.append(opt)
        # return as voronoi channel indices
        opts_indices = []
        for opt in opts:
            idx = m_inds[opt]
            opts_indices.append(self._get_channel((x[idx],y[idx])))
        return opts_indices

    def _get_crystal_opts_add(self, b_next_opts):
        crystals_opts = []
        masks_opts = []
        # fresh
        if len(self.crystals_curr)==0:
            for bv in b_next_opts:
                m = self._get_mask((bv))
                if not self._mask_is_already_present(m,self._mask_curr+masks_opts):
                    crystals_opts.append((bv))
                    masks_opts.append(m)
        else:
            # try adding new 1D crystals
            for bv in b_next_opts:
                m = self._get_mask((bv))
                if not self._mask_is_already_present(m,self._mask_curr+masks_opts):
                    crystals_opts.append((bv))
                    masks_opts.append(m)
            # try making an existing 1D crystal a 2D crystal
            for xtal in self._crystals_curr:
                if len(xtal)==1:
                    for bv in b_next_opts:
                        m = self._get_mask((xtal[0],bv))
                        if not self._mask_is_already_present(m,self._mask_curr+masks_opts):
                            crystals_opts.append((xtal[0],bv))
                            mask_opts.append(m)
        # return
        return crystal_opts,mask_opts


    #def _get_crystal_opts_rm(self, bvs_curr):


    def _score_crystal_options_and_update(self,crystals_opts,mask_opts):
        """
        """
        for xtals,mask in zip(crystals_opts,mask_opts):
            cost = self._get_cost(xtals,mask)
            if cost > self._best_score_curr:
                self._best_crystal_set_curr = xtals
                self._mask_curr = mask
                self._best_score_curr = cost

    def _get_cost(self, xtals, mask):
        """ The cost is
                f = -1e6a-1e3b-c
        where
            a = # unmasked data points
            b = # basis vectors
            c = # on-axis crystals
        """
        a = len(set(self._data_channels_curr)-set(mask))
        b = 0
        c = 0
        for xtal in xtals:
            l = len(xtal)
            b += l
            if l>1:
                c += 1
        return -1e6*a-1e3*b-c


    def _get_mask(self,crystal):
        """ crystal should be a tuple of 1 or 2 ints (off/on-axis) indicating basis vectors
        as indices in the voronoi.points
        """
        l = len(crystal)
        if l == 1:
            return self._get_mask_1D(crystal(0))
        elif l == 2:
            return self._get_mask_2D(crystal(0),crystal(1))
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

    def _mask_is_already_present(self, mask, masks):
        m = set(mask)
        for _m in masks:
            if m == set(_m):
                return True
        return False


    def _combine_masks(self,masks):
        """ masks should be a list of masks
        """

    def _get_crystal_permutations(self, bs):
        """
        Accepts a list of basis vectors `bs`...
        """
        # TODO
        pass

    def _get_score(self,mask,b,c):
        """ b is the #basis vectors, c is b-#crystals (=#on-axis crystals)
        """
        pass

    def _get_scores(self,masks,bs,cs):
        """ batch for _get_score
        """
        pass








    def _seeded_path(self,seed):
        pass

    def _new_pixel(self,seed):
        pass

    def _seeded_pixel(self,seed):
        pass

    def _walk(self):
        pass

    def _turn(self):
        pass

    def _backtrack_and_spawn(self):
        pass

    def _clean_path(self):
        pass

    def _endloop(self):
        # TODO
        # wrap in an access class?
        pass

    def show_results(self):
        pass



    ### Utilities

    def _get_b_next(self,x,y,inten):
        # TODO
        pass

    def _show_voronoi_mask(self,mask,mask_alpha=0.72,mask_color='y',
        c='w',lw=1,vp={},returnfig=False):
        """ mask is a list of integer (voronoi regions)
        """
        patches = []
        fig,ax = self.show_voronoi(c=c,lw=lw,vp=vp,returnfig=True)
        for idx in mask:
            vertices_curr = self._voronoi_vertices[idx]
            vert = np.roll(vertices_curr,-1,1)
            patches.append(Polygon(vert))
        p = PatchCollection(patches,alpha=mask_alpha,color=mask_color)
        ax.add_collection(p)
        if returnfig:
            return fig,ax
        else:
            plt.show()

### Utility classes

class ACCHOOPath:

    def __init__(self,seed):
        self.seed = seed
        self.path_pos = 0
        self.path_directions = []
        self.path_indices = []

