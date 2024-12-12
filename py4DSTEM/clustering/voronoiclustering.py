import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import get_cmap
from numpy.linalg import lstsq
from itertools import permutations
from scipy.spatial import Voronoi
from skimage.measure import label
from sklearn.decomposition import NMF
from scipy.ndimage import gaussian_filter
from scipy.ndimage import (gaussian_filter,binary_opening,binary_closing,
    binary_dilation,binary_erosion,)
from py4DSTEM.braggvectors import BraggVectors
from py4DSTEM.utils import get_maxima_2D, get_voronoi_vertices
from py4DSTEM.visualize import show, show_points

class VoronoiClustering(object):
    """
    Cluster data into sets of pixels containing distinct crystals using the
    Voronoi partitioning methods described in Savitzky et al. "py4DSTEM:
    A software package for four-dimensional scanning transmission electron
    microscopy data analysis." Microscopy and Microanalysis 27.4, 712-743 (2022).

    A Voronoi diagram or tesselation is a partitioning of space in which a
    set of seed points are specified then space is divided into regions
    each of which is the set of points closest to a seed point. In this
    algorithm, maxima in the bragg vector map (2D histogram of all vector Bragg
    intensities across the dataset) are identified and used to seed a Voronoi
    partition of diffraction space, which are the algorithms channels.

    This class includes a heuristic to identify a rough initial guess,
    a matrix decomposition method to smooth the edges, visualization
    tools, and methods to manually or algorithmically merge or divide
    classes.

    Usage:

    >>> v = VoronoiClustering()
    >>> v.get_kpoints(...)
    >>> ...
    """
    # initialize
    def __init__(self, braggvectors, upsample=1, X_is_boolean=True, max_dist=None, verbose=True):
        """
        Parameters
        ----------
        braggvectors : BraggVectors
        upsample : integer
            upsample factor for the Bragg vector histogram
        X_is_boolean : bool
            treat braggvectors as booleans, ignoring intensities
        max_dist : None or number
            maximum allowable distance from a voronoi region seed for
            a peak to be included in that region
        """
        assert isinstance(
            braggvectors, BraggVectors
        ), "braggvectors must be a BraggVectors instance"
        self._setup_braggvectors(braggvectors,upsample)
        self._X_is_boolean = X_is_boolean
        self.max_dist = max_dist
        self._v = verbose
        return

    def _setup_braggvectors(self, braggvectors, upsample=1):
        self.b = self.braggvectors = braggvectors
        self.upsample = upsample
        self.bvm = self.b.histogram(sampling=upsample)

    # Set data channels
    def get_kpoints(self, p, vp={}, show=True, update_data_matrix=True):
        """
        Select maxima in the bragg vector map to serve as voronoi region seeds,
        i.e. defining the algorithm's partitioning of the diffraction plane.
        This method then determines the data channels present at each scan
        position and constructs the data matrix.

        Parameters
        ----------
        p : dict
            parameter dict to pass to get_maxima_2D; see that methods docstring
            (in py4DSTEM.utils)
        vp : dict
            parameter dict to pass to show when visualizing the results
        get_data_matrix : bool
            toggles updating the data channels and data matrix
        """
        # find the points
        ans = get_maxima_2D(self.bvm.data, **p)
        self._qx = ans['x']
        self._qy = ans['y']
        self._int = ans['intensity']
        # transform coordinates
        origin = self.b.calibration.get_origin_mean()
        self.origin = np.array([origin[0]*self.upsample, origin[1]*self.upsample])
        self.qx = self._qx - self.origin[0]
        self.qy = self._qy - self.origin[1]
        self.qx *= self.qpixsize
        self.qy *= self.qpixsize
        # store convenience objects
        self._points = np.vstack((self.qx,self.qy)).T
        s = self.b.Qshape
        self.FOV = (s[0]*self.qpixsize*self.upsample, s[1]*self.qpixsize*self.upsample)
        self.Lmax = np.hypot(self.FOV[0],self.FOV[1])/2
        fov = (self.FOV[0]*1.05, self.FOV[1]*1.05)
        self._FOV_centered = (-fov[0]/2, fov[0]/2, -fov[1]/2, fov[1]/2)
        # get voronoi tesselation
        self._voronoi_points = np.vstack((self._qx,self._qy)).T
        self._voronoi = Voronoi(self._voronoi_points)
        self._voronoi_vertices = get_voronoi_vertices(
            self._voronoi, self.b.Qshape[0], self.b.Qshape[1])
        # show
        print(f"Identified {len(ans)} diffraction maxima; setting up voronoi channel data basis.")
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
        if update_data_matrix:
            # get the braggpeak labels
            self.get_braggpeak_labels()
            # construct X matrix
            self.X = np.zeros((self.N_feat, self.N_meas))
            for rx in range(self.R_Nx):
                for ry in range(self.R_Ny):
                    r = rx*self.R_Ny + ry
                    s = self.peaklabels[rx][ry]
                    p = self.b.cal[rx,ry]
                    for i in s:
                        if self._X_is_boolean:
                            self.X[i, r] = True
                        else:
                            d = np.hypot(
                                p["qx"]-self.qx[i],
                                p["qy"]-self.qy[i],
                            )
                            ind = np.argmin(d)
                            self.X[i, r] = p["intensity"][ind]

    def get_braggpeak_labels(self):
        """ Gets the set of integers specifying the bragg peaks' voronoi regions
        at each scan position.
        """
        # setup data structure
        braggpeak_labels = [
            [set() for i in range(self.b.shape[1])] \
                for j in range(self.b.shape[0])]
        # loop and find indices
        for rx in range(self.shape[0]):
            for ry in range(self.shape[1]):
                s = braggpeak_labels[rx][ry]
                p = self.b.cal[rx,ry]
                for i in range(len(p.data)):
                    d = np.hypot(self.qx-p.qx[i], self.qy-p.qy[i])
                    label = np.argmin(d)
                    # filter by distance if requested
                    if self.max_dist is not None and d[label] < self.max_dist:
                        s.add(label)
                    else:
                        s.add(label)
        # store and exit
        self.peaklabels = braggpeak_labels
        pass

    def voronoi_cooccurence_cluster(self,thresh=0.3,frac_thresh=0.1,max_iters=200,
        n_corr_init=2,):
        """
        Makes an initial guess at the classes present in the dataset using
        the voronoi coccurence heuristic described in Savitzky et al. "py4DSTEM:
        A software package for four-dimensional scanning transmission electron
        microscopy data analysis." Microscopy and Microanalysis 27.4, 712-743
        (2024). The algorithm is below. Throughout, BP or peak refer to the
        voronoi channel indices.

        Algo
        ----
        1. Calculate an n-point correlation function, i.e. the joint probability
        of any given n BPs coexisting in a diffraction pattern.  n is controlled
        by n_corr_init, and must be 2 or 3. E.g. the 3-point probablity
        P(x,y;i,j) = P(peaks i,j,k are all in the DP at (x,y)).
        2. Find the BP set maximizing the probability and call this a new class
        3. Find all DPs containing the class peaks. Calculate the next most
        probably peak to also be present and, if its probability of cooccuring
        with class thus far is greater than thresh, add it to the class and
        repeat this step. Otherwise, proceed to the next step.
        4. Check for the an condition and if met, store the classes and finish.
        Otherwise, set all slices in the correlation function containing the
        channels in this class to zero and go to step 2. The end condition is
        either frac_thresh of the voronoi channels are already included in a
        class, or, max_iters have been reached.
        5. Use these classes to construct an initial guess at a factorization
        of the data matrix X.  X is 2D with shape (CH,POS) where CH is the
        number of voronoi channels and POS is the number of scan positions.
        If WH = X then let W & H have shapes (CH,CLA) & (CLA,POS) where CLA
        is the number of classes.  Construct W by letting each of the CLA
        1D column vectors of len CH be 1 if this voronoi channel is in this
        class and 0 otherwise. H is then the leastsquare solution to the
        factorization, followed by setting all negative values to 0.

        Parameters
        ----------
        thresh : float in [0,1]
            threshold for adding new BPs to a class
        frac_thresh : float in [0,1]
            algorithm terminates if fewer than this fraction of the BPs have not
            been assigned to a class
        max_iters : int
            algorithm terminates after this many iterations
        n_corr_init : int
            seed new classes by finding maxima of the n-point joint probability
            function.  Must be 2 or 3.
        """
        if self._v:
            print('Commencing voronoi cooccurence cluster...')
            print(f'Using the {n_corr_init}-fold joint probability...')
        assert thresh >= 0 and thresh <= 1
        assert frac_thresh >= 0 and frac_thresh <= 1
        assert n_corr_init in (2, 3)
        N = self.N_feat
        R_Nx,R_Ny = self.R_Nx,self.R_Ny
        # two point probability
        if n_corr_init == 2:
            # Get two-point function
            n_point_function = np.zeros((N, N))
            for Rx in range(R_Nx):
                for Ry in range(R_Ny):
                    s = self.peaklabels[Rx][Ry]
                    perms = permutations(s, 2)
                    for perm in perms:
                        n_point_function[perm[0], perm[1]] += 1
            n_point_function /= R_Nx * R_Ny
            # Main loop
            BP_sets = []
            iteration = 0
            unused_BPs = np.ones(N, dtype=bool)
            seed_new_class = True
            while seed_new_class:
                ind1, ind2 = np.unravel_index(np.argmax(n_point_function), (N, N))
                BP_set = set([ind1, ind2])
                grow_class = True
                while grow_class:
                    frequencies = np.zeros(N)
                    N_elements = 0
                    for Rx in range(R_Nx):
                        for Ry in range(R_Ny):
                            s = self.peaklabels[Rx][Ry]
                            if BP_set.issubset(s):
                                N_elements += 1
                                for i in s:
                                    frequencies[i] += 1
                    frequencies /= N_elements
                    for i in BP_set:
                        frequencies[i] = 0
                    ind_new = np.argmax(frequencies)
                    if frequencies[ind_new] > thresh:
                        BP_set.add(ind_new)
                    else:
                        grow_class = False
                # Modify 2-point function, add new BP set to list, and decide to continue or stop
                for i in BP_set:
                    n_point_function[i, :] = 0
                    n_point_function[:, i] = 0
                    unused_BPs[i] = 0
                for s in BP_sets:
                    if len(s) == len(s.union(BP_set)):
                        seed_new_class = False
                if seed_new_class is True:
                    BP_sets.append(BP_set)
                iteration += 1
                N_unused_BPs = np.sum(unused_BPs)
                if iteration > max_iters or N_unused_BPs < N * frac_thresh:
                    seed_new_class = False
        # three point joint probability
        else:
            # Get three-point function
            n_point_function = np.zeros((N, N, N))
            for Rx in range(R_Nx):
                for Ry in range(R_Ny):
                    s = self.peaklabels[Rx][Ry]
                    perms = permutations(s, 3)
                    for perm in perms:
                        n_point_function[perm[0], perm[1], perm[2]] += 1
            n_point_function /= R_Nx * R_Ny
            # Main loop
            BP_sets = []
            iteration = 0
            unused_BPs = np.ones(N, dtype=bool)
            seed_new_class = True
            while seed_new_class:
                ind1, ind2, ind3 = np.unravel_index(np.argmax(n_point_function), (N, N, N))
                BP_set = set([ind1, ind2, ind3])
                grow_class = True
                while grow_class:
                    frequencies = np.zeros(N)
                    N_elements = 0
                    for Rx in range(R_Nx):
                        for Ry in range(R_Ny):
                            s = self.peaklabels[Rx][Ry]
                            if BP_set.issubset(s):
                                N_elements += 1
                                for i in s:
                                    frequencies[i] += 1
                    frequencies /= N_elements
                    for i in BP_set:
                        frequencies[i] = 0
                    ind_new = np.argmax(frequencies)
                    if frequencies[ind_new] > thresh:
                        BP_set.add(ind_new)
                    else:
                        grow_class = False
                # Modify 3-point function, add new BP set to list, and decide to continue or stop
                for i in BP_set:
                    n_point_function[i, :, :] = 0
                    n_point_function[:, i, :] = 0
                    n_point_function[:, :, i] = 0
                    unused_BPs[i] = 0
                for s in BP_sets:
                    if len(s) == len(s.union(BP_set)):
                        seed_new_class = False
                if seed_new_class is True:
                    BP_sets.append(BP_set)
                iteration += 1
                N_unused_BPs = np.sum(unused_BPs)
                if iteration > max_iters or N_unused_BPs < N * frac_thresh:
                    seed_new_class = False
        # store the answer
        self.BP_sets = BP_sets
        # populate the classes in the W and H matrices
        self.N_c = len(BP_sets)
        # W
        self.W = np.zeros((self.N_feat, self.N_c))
        for i in range(self.N_c):
            BP_set = BP_sets[i]
            for j in BP_set:
                self.W[j, i] = 1
        # H
        self.H = lstsq(self.W, self.X, rcond=None)[0]
        self.H = np.where(self.H < 0, 0, self.H)
        self.W_next = None
        self.H_next = None
        self.N_c_next = None
        if self._v:
            print(f"Done. Found {self.N_c} classes.")
        pass

    def get_initial_classes_from_images(self, images):
        """
        Populate the initial classes with a set of class images.

        Parameters
        ----------
        images : ndarray
            must have shape (R_Nx,R_Ny,N_c), where N_c will be the number of
            classes
        """
        assert images.shape[1] == self.R_Nx
        assert images.shape[2] == self.R_Ny
        # Construct W, H matrices
        self.N_c = images.shape[0]
        # H
        H = np.zeros((self.N_c, self.N_meas))
        for i in range(self.N_c):
            H[i, :] = images[i, :, :].ravel()
        self.H = np.copy(H, order="C")
        # W
        W = lstsq(self.H.T, self.X.T, rcond=None)[0].T
        W = np.where(W < 0, 0, W)
        self.W = np.copy(W, order="C")
        self.W_next = None
        self.H_next = None
        self.N_c_next = None
        pass

    def nmf(self, max_iterations=1):
        """
        Nonnegative matrix factorization using the scikit-learn NMF class.
        The current class guesses are refined by iteratively attempting to
        solve X = WH, factoring the data X into two smaller matrices W and
        H, where

            * X is the data matrix. It has shape (N_feat,N_meas) where N_feat is
              the number of voronoi channels and N_meas is the number of scan
              positions. Element X[i,j] represents the value of the i'th BP in
              the j'th DP and will be boolean if the X_is_boolean flag is set and
              otherwise is the intensity of the peak found in this channel or 0.
            * W is the class matrix. It has shape (N_feat,N_c) where N_c is the
              number of classes. The i'th column vector, w_i = W[:,i], gives the
              weight of each Bragg peak in the i'th class and has length N_feat.
              The j'th row vector w_j = W[j,:] is how strongly the j'th BP is
              associated with the i'th class.
            * H is the coefficient matrix. It has shape (N_c,N_meas).  The i'th
              column vector H[:,i] describes the contribution of each class to
              scan position i and the j'th row vector h_j = H[j,:] is the weights
              of the j'th class at each pixel of the scan.  h_j is an unraveled
              class image.

        under the constraint that all elements of X, W and H are positive.

        Using NMF here is saying: we believe the data may be well represented
        by decomposition into a part made of partitions of diffraction space
        and a part made images. In so doing it reduces the dimensionality in
        terms of a basis which is comparatively small but effectively
        efficiently captures the data's information content. In this
        construction, a voronoi channel basis W and set of images H are used
        to deconstruct dataset X. Nonnegativity is physical, and may be thought
        of as requiring classes that accumulate, but can't remove, diffraction
        space intensity.

        Parameters
        ----------
        max_iterations : int
            the maximum number of NMF steps to take
        """
        sklearn_nmf = NMF(n_components=self.N_c, init="custom", max_iter=max_iterations)
        self.W_next = sklearn_nmf.fit_transform(self.X, W=self.W, H=self.H)
        self.H_next = sklearn_nmf.components_
        self.N_c_next = self.W_next.shape[1]
        pass

    def split(self, sigma=2, thresh=0.25, expand_mask=1, minimum_pixels=1):
        """
        If any classes contain multiple non-contiguous segments in real space,
        divide these regions into distinct classes. The algorithm is below

        Retrieve a class image, convolve with a gaussian of std sigma, then
        turn into a binary mask by applying a threshold of thresh. Eliminate
        single pixels with a binary opening then closing, then expand the mask
        by expand_mask pixels. Finally, the contiguous regions of the resulting
        mask are found and identified as new, distinct classes. Iterate over
        all classes.

        The new classes are new columns in W and have exactly the same channel
        values as the old classes. The two new rows in H represent images of
        the new, nonoverlapping images.

        Parameters
        ----------
        sigma : float
            std of gaussian kernel used to smooth the class images before
            thresholding and splitting.
        thresh : float
            used to threshold the class image to create a binary mask.
        expand_mask : int
            number of pixels by which to expand the mask before separating
            into contiguous regions.
        minimum_pixels : int
            if, after splitting, a potential new class contains fewer than
            this number of pixels, ignore it
        """
        assert isinstance(expand_mask, (int, np.integer))
        assert isinstance(minimum_pixels, (int, np.integer))
        # setup
        W_next = np.zeros((self.N_feat, 1))
        H_next = np.zeros((1, self.N_meas))
        # loop
        for i in range(self.N_c):
            # get the class in real space
            class_image = self.get_class_image(i)
            # turn into a binary mask
            class_image = gaussian_filter(class_image, sigma)
            mask = class_image > (np.max(class_image) * thresh)
            mask = binary_opening(mask, iterations=1)
            mask = binary_closing(mask, iterations=1)
            mask = binary_dilation(mask, iterations=expand_mask)
            # get connected regions
            labels, nlabels = label(mask, background=0, return_num=True, connectivity=2)
            # add each region to the new W and H matrices
            for j in range(nlabels):
                mask = labels == (j + 1)
                mask = binary_erosion(mask, iterations=expand_mask)
                if np.sum(mask) >= minimum_pixels:
                    # leave the Bragg peak weightings the same
                    W_next = np.hstack((W_next, self.W[:, i, np.newaxis]))
                    # use the existing real space pixel weightings
                    h_i = np.zeros(self.N_meas)
                    h_i[mask.ravel()] = self.H[i, :][mask.ravel()]
                    H_next = np.vstack((H_next, h_i[np.newaxis, :]))
        # update and exit
        self.W_next = W_next[:, 1:]
        self.H_next = H_next[1:, :]
        self.N_c_next = self.W_next.shape[1]
        pass

    def merge(self, thresh_bragg=0.1, thresh_image=0.1, iterate=True):
        """
        If any classes contain sufficient overlap in both scan positions and
        BPs, merge them into a single class.

        Algo
        ----
        Calculate the Pearson correlation coefficient matrix for the class
        Bragg peak representation (correlations of columns of W) and image
        representation (H rows). Class pairs with correlation coefficients
        exceeding either of the thresholds are marked as merge candidates.
        Merges are then performed in order descending coefficients, and no
        class is merged more than once to avoid errors in intransitive
        cases. By default the algorithm then loops until no more candidates
        satisfying the thresholds remain. To perform class merges, the new
        class column in W is created by adding the two class columns weighting
        by the class's total image space intensity and the new row in H by sum.

        Parameters
        ----------
        thresh_bragg : float
            the threshold for the bragg peaks correlation coefficient, above
            which the two classes are considered candidates for merging
        thresh_image : float
            the threshold for the scan position correlation coefficient
            above which two classes are considered merge candidates
        iterate : bool
            toggles iteration
        """
        # setup
        proceed = True
        W_ = np.copy(self.W)
        H_ = np.copy(self.H)
        Nc_ = W_.shape[1]
        # loop
        while proceed:
            # Get correlation coefficients
            W_corr = np.corrcoef(W_.T)
            H_corr = np.corrcoef(H_)
            # Get merge candidate pairs
            mask_BPs = W_corr > thresh_bragg
            mask_ScanPosition = H_corr > thresh_image
            mask_upperright = np.zeros((Nc_, Nc_), dtype=bool)
            for i in range(Nc_):
                mask_upperright[i, i + 1 :] = 1
            merge_mask = mask_BPs * mask_ScanPosition * mask_upperright
            merge_i, merge_j = np.nonzero(merge_mask)
            # Sort merge candidate pairs
            merge_candidates = np.zeros(
                len(merge_i),
                dtype=[
                    ("i", int),
                    ("j", int),
                    ("cc_w", float),
                    ("cc_h", float),
                    ("score", float),
                ],
            )
            merge_candidates["i"] = merge_i
            merge_candidates["j"] = merge_j
            merge_candidates["cc_w"] = W_corr[merge_i, merge_j]
            merge_candidates["cc_h"] = H_corr[merge_i, merge_j]
            merge_candidates["score"] = (
                W_corr[merge_i, merge_j] * H_corr[merge_i, merge_j])
            merge_candidates = np.sort(merge_candidates, order="score")[::-1]
            # Perform merge
            merged = np.zeros(Nc_, dtype=bool)
            W_merge = np.zeros((self.N_feat, 1))
            H_merge = np.zeros((1, self.N_meas))
            for index in range(len(merge_candidates)):
                i = merge_candidates["i"][index]
                j = merge_candidates["j"][index]
                if not (merged[i] or merged[j]):
                    weight_i = np.sum(H_[i, :])
                    weight_j = np.sum(H_[j, :])
                    W_new = (W_[:, i] * weight_i + W_[:, j] * weight_j) / (
                        weight_i + weight_j
                    )
                    H_new = H_[i, :] + H_[j, :]
                    W_merge = np.hstack((W_merge, W_new[:, np.newaxis]))
                    H_merge = np.vstack((H_merge, H_new[np.newaxis, :]))
                merged[i] = True
                merged[j] = True
            # update
            W_merge = W_merge[:, 1:]
            H_merge = H_merge[1:, :]
            W_ = np.hstack((W_[:, merged == False], W_merge))  # noqa: E712
            H_ = np.vstack((H_[merged == False, :], H_merge))  # noqa: E712
            Nc_ = W_.shape[1]
            # continue?
            if len(merge_candidates)==0 or not(iterate):
                proceed = False
        # update variables and exit
        self.W_next = W_
        self.H_next = H_
        self.N_c_next = self.W_next.shape[1]
        pass

    def merge_ij(self, i, j):
        """
        Merge classes i and j into a single class. The new class column in W
        is generated by adding the two class columns weighting by the class's
        total real space intensity and the row in H by simple sum.
        """
        assert np.all(
            [isinstance(ind, (int, np.integer)) for ind in [i, j]]
        ), "i and j must be ints"

        # Get merged class
        weight_i = np.sum(self.H[i, :])
        weight_j = np.sum(self.H[j, :])
        W_new = (self.W[:, i] * weight_i + self.W[:, j] * weight_j) / (
            weight_i + weight_j
        )
        H_new = self.H[i, :] + self.H[j, :]
        # remove old classes, add new class, and exit
        self.W_next = np.delete(self.W, j, axis=1)
        self.H_next = np.delete(self.H, j, axis=0)
        self.W_next[:, i] = W_new
        self.H_next[i, :] = H_new
        self.N_c_next = self.N_c - 1
        return

    def split_i(self,i,sigma=2,thresh=0.25,expand_mask=1,
        minimum_pixels=1):
        """
        If class i contains multiple non-contiguous segments in real space,
        divide these regions into distinct classes. See the self.split
        docstring for the algorithm.

        Parameters
        ----------
        i : int
            index of the class to split
        sigma : float
            std of gaussian kernel used to smooth the class images before
            thresholding and splitting.
        thresh : float
            used to threshold the class image to create a binary mask.
        expand_mask : int
            number of pixels by which to expand the mask before
            separating into contiguous regions.
        minimum_pixels : int
            if, after splitting, a potential new class contains
            fewer than this number of pixels, ignore it
        """
        assert isinstance(i, (int, np.integer))
        assert isinstance(expand_mask, (int, np.integer))
        assert isinstance(minimum_pixels, (int, np.integer))
        W_next = np.zeros((self.N_feat, 1))
        H_next = np.zeros((1, self.N_meas))
        # Get the class in real space
        class_image = self.get_class_image(i)
        # Turn into a binary mask
        class_image = gaussian_filter(class_image, sigma)
        mask = class_image > (np.max(class_image) * thresh)
        mask = binary_opening(mask, iterations=1)
        mask = binary_closing(mask, iterations=1)
        mask = binary_dilation(mask, iterations=expand_mask)
        # Get connected regions
        labels, nlabels = label(mask, background=0, return_num=True, connectivity=2)
        # Add each region to the new W and H matrices
        for j in range(nlabels):
            mask = labels == (j + 1)
            mask = binary_erosion(mask, iterations=expand_mask)
            if np.sum(mask) >= minimum_pixels:
                # Leave the Bragg peak weightings the same
                W_next = np.hstack((W_next, self.W[:, i, np.newaxis]))
                # Use the existing real space pixel weightings
                h_i = np.zeros(self.N_meas)
                h_i[mask.ravel()] = self.H[i, :][mask.ravel()]
                H_next = np.vstack((H_next, h_i[np.newaxis, :]))
        # update and exit
        W_prev = np.delete(self.W, i, axis=1)
        H_prev = np.delete(self.H, i, axis=0)
        self.W_next = np.concatenate((W_next[:, 1:], W_prev), axis=1)
        self.H_next = np.concatenate((H_next[1:, :], H_prev), axis=0)
        self.N_c_next = self.W_next.shape[1]
        return

    def remove_i(self, i):
        """ Remove class i.
        """
        assert isinstance(i, (int, np.integer))
        self.W_next = np.delete(self.W, i, axis=1)
        self.H_next = np.delete(self.H, i, axis=0)
        self.N_c_next = self.W_next.shape[1]
        return

    def accept(self):
        """ Update the W and H matrices with the current candidate classes.
        """
        if self.W_next is None or self.H_next is None:
            return
        else:
            self.W = self.W_next
            self.H = self.H_next
            self.N_c = self.N_c_next
            self.W_next = None
            self.H_next = None
            self.N_c_next = None

    def reject(self):
        """ Discard the current candidate classes.
        """
        self.W_next = None
        self.H_next = None
        self.N_c_next = None

    def get_class(self, i):
        """ Get class i's voronoi channel weights and image (scan position
        weights), returing them as a 2-tuple.
        """
        class_BPs = self.W[:, i]
        class_image = self.H[i, :].reshape((self.R_Nx, self.R_Ny))
        return class_BPs, class_image

    def get_class_BPs(self, i):
        """ Get class i, returning its voronoi channel weights.
        """
        return self.W[:, i]

    def get_class_image(self, i):
        """ Get class i's image, returning its scan position weights.
        """
        return self.H[i, :].reshape((self.R_Nx, self.R_Ny))

    def set_class_image(self, i, im):
        """ Set class i's image, acceting an (Rx,Ry) shaped array
        """
        assert(im.shape==(self.R_Nx,self.R_Ny))
        self.H[i, :] = im.ravel()

    def get_candidate_class(self, i):
        """ Get candidate class i's voronoi channel weights and image (scan
        position weights), returing them as a 2-tuple.
        """
        assert self.W_next is not None, "W_next is not assigned."
        assert self.H_next is not None, "H_next is not assigned."
        class_BPs = self.W_next[:, i]
        class_image = self.H_next[i, :].reshape((self.R_Nx, self.R_Ny))
        return class_BPs, class_image

    def get_candidate_class_BPs(self, i):
        """ Get candidate class i, returning its voronoi channel weights.
        """
        assert self.W_next is not None, "W_next is not assigned."
        return self.W_next[:,i]

    def get_candidate_class_image(self,i):
        """ Get candidate class i's image, returning its scan position weights.
        """
        assert self.H_next is not None, "H_next is not assigned."
        return self.H_next[i,:].reshape((self.R_Nx,self.R_Ny))

    # properties
    @property
    def shape(self):
        return self.braggvectors.shape
    @property
    def Qshape(self):
        return self.braggvectors.Qshape
    @property
    def R_Nx(self):
        return self.shape[0]
    @property
    def R_Ny(self):
        return self.shape[1]
    @property
    def Q_Nx(self):
        return self.Qshape[0]
    @property
    def Q_Ny(self):
        return self.Qshape[1]
    @property
    def qpixsize(self):
        return self.b.calibration.get_Q_pixel_size()/self.upsample
    @property
    def N_feat(self):
        """ the number of bragg peaks """
        return len(self.qx)
    @property
    def N_meas(self):
        """ the number of scan positions """
        return self.R_Nx*self.R_Ny

    # visualization
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
        vp={},verbose=False,returnfig=False):
        """ show the data points, mask, voronoi, bvm overlaid
        """
        rx,ry = coord
        mask = self.state_masks[rx][ry]
        # show voronoi diagram
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
            crystal_inds = self.state_crystals[coord[0]][coord[1]]
            crystals = [self.crystals[idx] for idx in crystal_inds]
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
            if verbose:
                print(f"crystal channels: {crystals}")
        # return
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_data_mask_DP_compare(
        self,
        coord,
        figsize=(10,5),
        vp={},
        c_scat='r',
        s_scat=25,
        sp={},
        c_vor='lightcyan',
        lw=0.5,
        mask_color='y',
        mask_alpha=0.4,
        marker='x',
        markercolor='blue',
        markersize=100,
        vectors=False,
        vect_cmap='cool',
        vect_width=0.5,
        vect_headsize=6,
        vectp={},
        dpp={'scaling':'log'},
        dp_rotate=0,
        dp_invert=False,
        returnfig=False):
        """ Full comparison including the diffraction data side-by-side

        Parameters
        ----------
        coord : tuple of ints
            the scan coordinate
        figsize : tuple
        vp : dict
            visualization parameters to pass to ax.show(bvm, **vp)
        c_scat : color
            color of voronoi maxima circles
        s_scat : number
            size of the voronoi maxima circles
        sp : dict
            vis params to pass to ax.scatter(..., **sp) for voronoi maxima circles
        c_vor : color
            color for voronoi edges
        lw : number
            linewidth for voronoi edges
        mask_color : color
            color for the voronoi mask
        mask_alpha : number
            transparency for the voronoi mask
        marker : str
            marker type for the data points
        markercolor : color
            color for the data points
        markersize : number
            size for the data points
        vectors : bool
            toggles showing crystal vectors that generated the mask
        vect_cmap : colormap
            the cmap used to draw vectors & differentiate if there are
            multiple crystals
        vect_width : number
            vector width
        vect_headsize : number
            vector headsize
        vectp : dict
            param dictionary to pass to add_vector(**vp)
        dpp : dict
            param dictionary to pass to show(diffraction_pattern, **dpp)
        dp_rotate : int
            rotates the diffraction pattern by 0=none,1=pi/2,2=pi,3=3pi/2
        dp_invert : bool
            inverst the diffraction pattern
        returnfig : bool
            toggles returning the figure
        """
        # ensure datacube is there
        assert(self._datacube is not None)
        # make the figure
        fig,(ax,ax2) = plt.subplots(1,2,figsize=figsize)
        # show the BVM
        show(self.bvm, figax=(fig,ax), **vp)
        # add voronoi maxima
        ax.scatter(self._qy, self._qx, s=s_scat, edgecolor=c_scat, facecolor="none", **sp)
        # add voronoi edges
        for region in range(len(self._voronoi_vertices)):
            vertices_curr = self._voronoi_vertices[region]
            if vertices_curr is not None:
                for i in range(len(vertices_curr)):
                    x0,y0 = vertices_curr[i,:]
                    x1,y1 = vertices_curr[(i+1)%len(vertices_curr),:]
                    ax.plot((y0,y1),(x0,x1),c_vor,lw=lw)
        ax.set_xlim([0,self.bvm.data.shape[1]])
        ax.set_ylim([0,self.bvm.data.shape[0]])
        ax.invert_yaxis()
        # get data and mask
        rx,ry = coord
        mask = self.state_masks[rx][ry]
        # show voronoi mask
        patches = []
        for idx in mask:
            vertices_curr = self._voronoi_vertices[idx]
            if vertices_curr is not None:
                vert = np.roll(vertices_curr,-1,1)
                patches.append(Polygon(vert))
        p = PatchCollection(patches,alpha=mask_alpha,color=mask_color)
        ax.add_collection(p)
        # transform data
        qpixsize = self.d.calibration.get_Q_pixel_size()
        origin = self.d.calibration.get_origin_mean()
        d = self.d.cal[rx,ry].data
        x,y = d['qx'],d['qy']
        x,y = self._transform_cal_to_pix(x,y)
        ax.scatter(y,x,color=markercolor,s=markersize,marker=marker)
        # if vectors were requested, add them
        if vectors:
            # set up vectors
            crystal_inds = self.state_crystals[coord[0]][coord[1]]
            crystals = [self.crystals[idx] for idx in crystal_inds]
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
        # show diffraction pattern
        dp = self._datacube[rx,ry]
        if dp_invert:
            dp = dp.T
        if dp_rotate!=0:
            dp = np.rot90(dp, k=dp_rotate)
        show(dp, figax=(fig,ax2), **dpp)
        # return
        if returnfig:
            return fig,(ax,ax2)
        else:
            plt.show()

    def show_clustering(self,thresh=0.3,cmap='hsv',show_current=True,
        figwidth=4,returnfig=False):
        """ Overlay of all class images.
        """
        if show_current:
            N_c = self.N_c
        else:
            N_c = self.N_c_next
        cmap_base = get_cmap(cmap)
        aspect_ratio = self.R_Ny/self.R_Nx
        fig,ax = plt.subplots(figsize=(figwidth,figwidth/aspect_ratio))
        ax.matshow(np.zeros((self.R_Nx,self.R_Ny)),cmap='gray')
        for index in range(N_c):
            if show_current:
                class_image = self.get_class_image(index)
            else:
                class_image = self.get_candidate_class_image(index)

            ma = np.ma.array(class_image, mask = class_image<thresh)
            if not np.all(ma.mask):
                colors = [(0,0,0,1),cmap_base(index/N_c)]
                cm = LinearSegmentedColormap.from_list('cmap', colors, N=100)
                ax.matshow(ma,cmap=cm)
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_class(self,index,thresh_bragg=0.5,colorA='r',colorB='y',markersize=50,
        show_current=True,figwidth=8,vp={},imp={},returnfig=False):
        """ Display a single class image and BP channels.

        Parameters
        ----------
        index : int
            class to display
        markersize : number
            size of the BP markers
        show_current : bool
            toggle displaying current vs. next state
        """
        # set up plot
        aspect_ratio1 = self.R_Ny/self.R_Nx
        aspect_ratio2 = self.Q_Ny/self.Q_Nx
        aspect_ratio = aspect_ratio1+aspect_ratio2
        fig,axs = plt.subplots(1,2,figsize=(figwidth,figwidth/aspect_ratio))
        ax1,ax2 = axs
        # get the classes
        if show_current:
            class_BPs, class_image = self.get_class(index)
        else:
            class_BPs, class_image = self.get_candidate_class(index)
        bps = class_BPs>thresh_bragg
        show(self.bvm.data,figax=(fig,ax1),**vp)
        ax1.scatter(self._qy,self._qx,edgecolor=colorA,facecolor='none',
            s=markersize,)
        ax1.scatter(self._qy[bps],self._qx[bps],color=colorB,
            s=markersize,)
        # show the image
        show(class_image,figax=(fig,ax2),**imp)
        # grid off
        ax1.grid(False)
        ax2.grid(False)
        # exit
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_classes(self,thresh_bragg=0.5,colorA='r',colorB='y',markersize=50,
        show_current=True,ncols=4,label=True,labeloffset=5,labelsize=24,
        labelcolor='w',stride=1,vp={},figwidth=8,imp={},returnfig=False):
        """
        """
        # get number of classes
        if show_current:
            N_c = self.N_c
        else:
            N_c = self.N_c_next
        # set up plot
        N_classes = N_c // stride
        nrows = int(np.ceil((N_classes)/ncols))
        aspect_ratio1 = self.R_Ny/self.R_Nx
        aspect_ratio2 = self.Q_Ny/self.Q_Nx
        aspect_ratio = aspect_ratio1+aspect_ratio2
        fig,axs = plt.subplots(nrows,2*ncols,figsize=(figwidth*ncols,figwidth*nrows/aspect_ratio))
        # loop over classes, get axes
        for index in range(N_classes):
            ax1 = axs[int(index//ncols),int(2*(index%ncols))]
            ax2 = axs[int(index//ncols),int(2*(index%ncols)+1)]
            # get the classes
            if show_current:
                class_BPs, class_image = self.get_class(index*stride)
            else:
                class_BPs, class_image = self.get_candidate_class(index*stride)
            bps = class_BPs>thresh_bragg
            show(self.bvm.data,figax=(fig,ax1),**vp)
            ax1.scatter(self._qy,self._qx,edgecolor=colorA,facecolor='none',
                s=markersize,)
            ax1.scatter(self._qy[bps],self._qx[bps],color=colorB,
                s=markersize,)
            # show the images
            show(class_image,figax=(fig,ax2),**imp)
            # add label
            if label:
                ax1.text(labeloffset,labeloffset+labelsize/2,"{}".format(index*stride),size=labelsize,color=labelcolor)
            # grid off
            ax1.grid(False)
            ax2.grid(False)
        # remove extra axes
        for index in range(N_classes,nrows*ncols):
            ax1 = axs[int(index//ncols),int(2*(index%ncols))]
            ax2 = axs[int(index//ncols),int(2*(index%ncols))+1]
            ax1.axis('off')
            ax2.axis('off')
        # exit
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_class_image(self,index,show_current=True,figwidth=4,imp={},
        returnfig=False):
        """ Display a single class image.

        Parameters
        ----------
        index : int
            class to display
        show_current : bool
            toggle displaying current vs. next state
        figwidth : number
            the figure width; height is autoscaled
        """
        # set up plot
        aspect_ratio = self.R_Ny/self.R_Nx
        fig,ax = plt.subplots(figsize=(figwidth,figwidth/aspect_ratio))
        # get the classes
        if show_current:
            class_image = self.get_class_image(index)
        else:
            class_image = self.get_candidate_class_image(index)
        # show the image
        show(class_image,figax=(fig,ax),**imp)
        # grid off
        ax.grid(False)
        # exit
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_class_images(self, show_current=True,ncols=4,label=True,labeloffset=5,
        labelsize=24,labelcolor='w',stride=1,vp={},figwidth=8,imp={},
        returnfig=False):
        """
        """
        # get number of classes
        if show_current:
            N_c = self.N_c
        else:
            N_c = self.N_c_next
        # set up plot
        N_classes = N_c // stride
        nrows = int(np.ceil((N_classes)/ncols))
        aspect_ratio = self.R_Ny/self.R_Nx
        fig,axs = plt.subplots(nrows,ncols,figsize=(figwidth*ncols,figwidth*nrows/aspect_ratio))
        # loop over classes, get axes
        for index in range(N_classes):
            ax = axs[int(index//ncols),int(index%ncols)]
            # get the classes
            if show_current:
                class_image = self.get_class_image(index*stride)
            else:
                class_image = self.get_candidate_class_image(index*stride)
            # show the images
            show(class_image,figax=(fig,ax),**imp)
            # add label
            if label:
                ax.text(labeloffset,labeloffset+labelsize/2,"{}".format(index*stride),size=labelsize,color=labelcolor)
            # grid off
            ax.grid(False)
        # remove extra axes
        for index in range(N_classes,nrows*ncols):
            ax = axs[int(index//ncols),int(index%ncols)]
            ax.axis('off')
        # exit
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_class_channel(self,index,thresh_bragg=0.5,colorA='r',colorB='y',markersize=50,
        show_current=True,figwidth=8,vp={},imp={},returnfig=False):
        """ Display a single class BP channels.

        Parameters
        ----------
        index : int
            class to display
        markersize : number
            size of the BP markers
        show_current : bool
            toggle displaying current vs. next state
        """
        # set up plot
        aspect_ratio = self.Q_Ny/self.Q_Nx
        fig,ax = plt.subplots(figsize=(figwidth,figwidth*aspect_ratio))
        # get the classes
        if show_current:
            class_BPs = self.get_class_BPs(index)
        else:
            class_BPs = self.get_candidate_class_BPs(index)
        bps = class_BPs>thresh_bragg
        show(self.bvm.data,figax=(fig,ax),**vp)
        ax.scatter(self._qy,self._qx,edgecolor=colorA,facecolor='none',
            s=markersize,)
        ax.scatter(self._qy[bps],self._qx[bps],color=colorB,
            s=markersize,)
        # grid off
        ax.grid(False)
        # exit
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_class_channels(self,thresh_bragg=0.5,colorA='r',colorB='y',
        markersize=50,show_current=True,ncols=4,label=True,labeloffset=5,
        labelsize=24,labelcolor='w',stride=1,vp={},figwidth=8,imp={},
        returnfig=False):
        """
        """
        # get number of classes
        if show_current:
            N_c = self.N_c
        else:
            N_c = self.N_c_next
        # set up plot
        N_classes = N_c // stride
        nrows = int(np.ceil((N_classes)/ncols))
        aspect_ratio = self.Q_Ny/self.Q_Nx
        fig,axs = plt.subplots(nrows,ncols,figsize=(figwidth*ncols,figwidth*nrows/aspect_ratio))
        # loop over classes, get axes
        for index in range(N_classes):
            ax = axs[int(index//ncols),int(index%ncols)]
            # get the classes
            if show_current:
                class_BPs = self.get_class_BPs(index*stride)
            else:
                class_BPs = self.get_candidate_class_BPs(index*stride)
            # show the class peaks
            bps = class_BPs>thresh_bragg
            show(self.bvm.data,figax=(fig,ax),**vp)
            ax.scatter(self._qy,self._qx,edgecolor=colorA,facecolor='none',
                s=markersize,)
            ax.scatter(self._qy[bps],self._qx[bps],color=colorB,
                s=markersize,)
            # add label
            if label:
                ax.text(labeloffset,labeloffset+labelsize/2,"{}".format(index*stride),size=labelsize,color=labelcolor)
            # grid off
            ax.grid(False)
        # remove extra axes
        for index in range(N_classes,nrows*ncols):
            ax = axs[int(index//ncols),int(index%ncols)]
            ax.axis('off')
        # exit
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_clustering_selected(self, indices, thresh=0.3, cmap='hsv', show_current=True,
        scalesize=4,figsize=(8,8),returnfig=True):
        """ Display class image overlays in N_c plots, adding classes one by one in
        each successive plot.

        Parameters
        ----------
        indices : list of ints
            classes to display
        thresh : number
            min display intensity
        cmap : colormap
        show_current : bool
            toggles showing the current vs. next state
        ncols : int
            number of display columns
        label : bool
            toggles display of the class index
        labeloffset : number
            label offset from the plot corner
        labelsize : number
            label size
        labelcolor : color
        stride : int
            display only every stride images
        scalesize : number
            scale the figure size
        figsize : 2-tuple
            figure size
        returnfig : bool
            toggle returning the figure
        """
        if show_current:
            N_c = self.N_c
        else:
            N_c = self.N_c_next
        cmap_base = get_cmap(cmap)
        fig,ax = plt.subplots(figsize=figsize)
        ax.matshow(np.zeros((self.R_Nx,self.R_Ny)),cmap='gray')
        for index in (indices):
            if show_current:
                class_image = self.get_class_image(index)
            else:
                class_image = self.get_candidate_class_image(index)

            ma = np.ma.array(class_image, mask = class_image<thresh)
            if not np.all(ma.mask):
                colors = [(0,0,0,1),cmap_base(index/N_c)]
                cm = LinearSegmentedColormap.from_list('cmap', colors, N=100)
                ax.matshow(ma,cmap=cm)
        # return
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_clustering_accum(self, thresh=0.3, cmap='hsv', show_current=True,
        ncols=1,label=True, labeloffset=5, labelsize=24, labelcolor='w', stride=1,
        scalesize=4,returnfig=False):
        """ Display class image overlays in N_c plots, adding classes one by one in
        each successive plot.

        Parameters
        ----------
        thresh : number
            min display intensity
        cmap : colormap
        show_current : bool
            toggles showing the current vs. next state
        ncols : int
            number of display columns
        label : bool
            toggles display of the class index
        labeloffset : number
            label offset from the plot corner
        labelsize : number
            label size
        labelcolor : color
        stride : int
            display only every stride images
        scalesize : number
            scale the figure size
        """
        if show_current:
            N_c = self.N_c
        else:
            N_c = self.N_c_next
        N_classes = N_c//stride
        ncols = int(ncols)
        nrows = int(np.ceil(N_classes/ncols))
        cmap_base = get_cmap(cmap)
        aspect_ratio = self.R_Ny/self.R_Nx
        fig,axs = plt.subplots(nrows,ncols,
            figsize=(scalesize*ncols,scalesize*nrows/aspect_ratio))
        for i in range(N_classes):
            if ncols==1:
                ax = axs[i]
            else:
                ax = axs[i//ncols,i%ncols]
            ax.matshow(np.zeros((self.R_Nx,self.R_Ny)),cmap='gray')
            for index in np.arange(i*stride+1):
                if show_current:
                    class_image = self.get_class_image(index)
                else:
                    class_image = self.get_candidate_class_image(index)

                ma = np.ma.array(class_image, mask = class_image<thresh)
                if not np.all(ma.mask):
                    colors = [(0,0,0,1),cmap_base(index/N_c)]
                    cm = LinearSegmentedColormap.from_list('cmap', colors, N=100)
                    ax.matshow(ma,cmap=cm)
            if label:
                ax.text(labeloffset,labeloffset+labelsize/2,"{}".format(index),size=labelsize,color=labelcolor)
        # remove excess axes
        for i in range(N_classes,ncols*nrows):
            ax = axs[i//ncols,i%ncols]
            ax.axis('off')
        # return
        if returnfig:
            return fig,axs
        else:
            plt.show()




















