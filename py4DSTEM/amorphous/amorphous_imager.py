# Define the AmorphousImager class, for calculating amorphous halo scattering intensity

import numpy as np
import matplotlib.pyplot as plt
from py4DSTEM.visualize import show
from py4DSTEM.process.polar import PolarDatacube
from py4DSTEM import tqdmnd
from scipy.optimize import curve_fit

class AmorphousImager:
    """
    Calculates an image of amorphous matter by finding the radial median curve, fitting a background to that curve
    about an amorphous halo at some q-range, then integrating the signal above that background.
    """
    def __init__(self, datacube, n_annular=180, qstep=0.5, fitfunction='exp', ymax=0.006):
        """
        """
        self.datacube = datacube
        self._setup_polarcube(n_annular,qstep)
        self.set_fitfunction(fitfunction)
        self.set_ymax(ymax)

    def _setup_polarcube(self, n_annular, qstep):
        """
        """
        self.polarcube = PolarDatacube(
            self.datacube,
            n_annular = n_annular,
            qstep = qstep
        )

    def set_fitfunction(self, f):
        """
        """
        assert(f in ('exp','poly'))
        self.fitfunction = f
        if self.fitfunction == 'exp':
            self.fitfun = self.fitfun_exp
        else:
            self.fitfun = self.fitfun_poly
        pass

    def set_ymax(self, ymax):
        """
        """
        self.ymax = ymax

    def setup_windows(self,pos,presignal,postsignal,signal,ymax=None,returnfig=False):
        """
        """
        self.set_fit_windows(presignal,postsignal)
        self.set_integration_window(signal)
        fig,ax = self.show_windows(pos[0],pos[1],ymax=ymax,returnfig=True)
        if returnfig:
            return fig,ax
        else:
            plt.show()
            pass

    def set_fit_windows(self,presignal,postsignal):
        """
        """
        self.presignal_window = presignal
        self.postsignal_window = postsignal
        # Setup data masks
        self.mask_fit_window0 = np.logical_and(
            self.polarcube.qq>self.presignal_window[0],
            self.polarcube.qq<=self.presignal_window[1]
        )
        self.mask_fit_window1 = np.logical_and(
            self.polarcube.qq>self.postsignal_window[0],
            self.polarcube.qq<=self.postsignal_window[1]
        )
        self.mask_fit = np.logical_or(self.mask_fit_window0,self.mask_fit_window1)
        pass

    def set_integration_window(self,window):
        """
        """
        self.integration_window = window
        # Setup integration mask
        self.mask_integration = np.logical_and(
            self.polarcube.qq>self.integration_window[0],
            self.polarcube.qq<=self.integration_window[1]
        )
        pass

    def show_windows(self,rx,ry,ymax=None,returnfig=False,legend=True):
        """
        """
        if ymax is not None:
            self.set_ymax(ymax)
        self._get_median_profiles(rx,ry)
        # Show
        fig,ax = plt.subplots()
        ax.plot(
            self.polarcube.qq,
            self.median_profile_localave,
            color = 'darkblue',
            label = 'median, local ave',
        )
        ax.plot(
            self.polarcube.qq,
            self.median_profile,
            color = 'lightblue',
            label = 'median',
        )
        ax.set_ylim(0,self.ymax)
        ax.set_xlim(0,.9)
        ax.axvspan(
            self.presignal_window[0],
            self.presignal_window[1],
            0,
            1000*self.ymax,
            color = 'y',
            alpha = 0.3,
            label = 'fit window',
        )
        ax.axvspan(
            self.postsignal_window[0],
            self.postsignal_window[1],
            0,
            1000*self.ymax,
            color = 'y',
            alpha = 0.3
        )
        ax.axvspan(
            self.integration_window[0],
            self.integration_window[1],
            0,
            1000*self.ymax,
            color = 'purple',
            alpha = 0.2,
            label = 'integration window',
        )
        if legend:
            plt.legend()
        if returnfig:
            return fig,ax
        else:
            plt.show()
            pass

    def _get_median_profiles(self,rx,ry):
        """
        """
        self.median_profile = np.ma.median(
            self.polarcube.data[rx,ry],
            axis=0
        )
        self.median_profile_localave = np.ma.median(
            self.polarcube.transform(
                self.datacube.get_local_ave_dp(rx,ry),
                origin = [rx,ry],
                ellipse = self.datacube.calibration.get_ellipse(),
            ),
            axis=0
        )
        return self.median_profile, self.median_profile_localave

    # Fit functions
    @staticmethod
    def fitfun_exp(x,c0,c1,c2):
        return c0 + c1*np.exp(-c2*x)
    @staticmethod
    def fitfun_poly(x,c0,c1,c2,c3,c4):
        return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
    @staticmethod
    def fitfun_line(x,c0,c1):
        return c0 + c1*x

    def fit_single_pattern(self,rx,ry,ymax=None):
        """
        """
        if ymax is not None:
            self.set_ymax(ymax)
        # Get profiles
        self._get_median_profiles(rx,ry)

        # Get fit data
        xvals_fit = self.polarcube.qq[self.mask_fit]
        yvals_fit = self.median_profile_localave[self.mask_fit]

        # Get initial guess at fit params
        x0 = np.mean(self.polarcube.qq[self.mask_fit_window0])
        y0 = np.mean(self.median_profile_localave[self.mask_fit_window0])
        x1 = np.mean(self.polarcube.qq[self.mask_fit_window1])
        y1 = np.mean(self.median_profile_localave[self.mask_fit_window1])

        # Fit - exponential
        if self.fitfunction == 'exp':
            try:
                fitfun = self.fitfun_exp
                c0_guess = y1*0.8
                y0 -= c0_guess
                y1 -= c0_guess
                c2_guess = -np.log(y1/y0)/(x1-x0)
                c1_guess = np.exp(-c2_guess*x0)/y0
                p0_guess = {
                    'c0' : c0_guess,
                    'c1' : c1_guess,
                    'c2' : c2_guess,
                }
                # Fit
                p0,pcov = curve_fit(
                    fitfun,
                    xdata = xvals_fit,
                    ydata = yvals_fit,
                    p0 = np.array([p0_guess[k] for k in ('c0','c1','c2')])
            )

            # If an exponential fit fails, fit a line
            except RuntimeError:
                fitfun = self.fitfun_line
                c0_guess = (y1-y0)/(x1-x0)
                c1_guess = y0 - c0_guess*x0
                p0_guess = {
                    'c0' : c0_guess,
                    'c1' : c1_guess,
                }
                # Fit
                p0,pcov = curve_fit(
                    fitfun,
                    xdata = xvals_fit,
                    ydata = yvals_fit,
                    p0 = np.array([p0_guess[k] for k in ('c0','c1')])
                    )

        # Fit - polynomial
        else:
            try:
                fitfun = self.fitfun_poly
                c0_guess = (y0 - y1*(x0/x1)**2)/(1 - (x0/x1)**2)
                c2_guess = (y1-c0_guess) / x1**2
                p0_guess = {
                    'c0' : c0_guess,
                    'c1' : 0,
                    'c2' : c2_guess,
                    'c3' : 0,
                    'c4' : 0
                }
                # Fit
                p0,pcov = curve_fit(
                    fitfun,
                    xdata = xvals_fit,
                    ydata = yvals_fit,
                    p0 = np.array([p0_guess[k] for k in ('c0','c1','c2','c3','c4')])
                )

            # If a polynomial fit fails, fit a line
            except RuntimeError:
                fitfun = self.fitfun_line
                c0_guess = (y1-y0)/(x1-x0)
                c1_guess = y0 - c0_guess*x0
                p0_guess = {
                    'c0' : c0_guess,
                    'c1' : c1_guess,
                }
                # Fit
                p0,pcov = curve_fit(
                    fitfun,
                    xdata = xvals_fit,
                    ydata = yvals_fit,
                    p0 = np.array([p0_guess[k] for k in ('c0','c1')])
                    )
        self._params_curr = fitfun, p0, pcov
        return self._params_curr

    def test_single_pattern_fit(self,rx,ry,ymax=None,returnfig=False,legend=True):
        ""
        ""
        if ymax is not None:
            self.set_ymax(ymax)
        fitfun, p0, pcov = self.fit_single_pattern(rx,ry)
        fig,ax = self.show_windows(rx,ry,returnfig=True,legend=False)
        ax.plot(
            self.polarcube.qq,
            fitfun(self.polarcube.qq,*p0),
            color = 'green',
            label = 'background',
        )
        mask_show_bksb = np.logical_and(
            self.polarcube.qq>self.presignal_window[1],
            self.polarcube.qq<=self.postsignal_window[0]
        )
        ax.plot(
            self.polarcube.qq[mask_show_bksb],
            self.median_profile[mask_show_bksb] - fitfun(self.polarcube.qq,*p0)[mask_show_bksb],
            color = 'red',
            label = 'amorphous',
        )
        if legend:
            plt.legend()
        if returnfig:
            return fig,ax
        else:
            plt.show()
            pass

    def select_test_patterns(self,rxs,rys,ymax=None,returnfig=False,legend=True):
        """
        """
        if ymax is not None:
            self.set_ymax(ymax)
        # figure
        fig,axs = plt.subplots(2,4,figsize=(12,6))
        # loop
        for idx,(rx,ry) in enumerate(zip(rxs,rys)):

            # get axis
            ax_i = idx//4
            ax_j = idx%4
            ax = axs[ax_i,ax_j]

            # Get median profiles
            self._get_median_profiles(rx,ry)

            # Show
            ax.plot(
                self.polarcube.qq,
                self.median_profile_localave,
                color = 'darkblue',
                label = 'median, local ave',
            )
            ax.plot(
                self.polarcube.qq,
                self.median_profile,
                color = 'lightblue',
                label = 'median',
            )
            ax.set_ylim(0,self.ymax)
            ax.set_xlim(0,.9)
            ax.axvspan(
                self.presignal_window[0],
                self.presignal_window[1],
                0,
                1000*self.ymax,
                color = 'y',
                alpha = 0.3,
                label = 'fit window',
            )
            ax.axvspan(
                self.postsignal_window[0],
                self.postsignal_window[1],
                0,
                1000*self.ymax,
                color = 'y',
                alpha = 0.3
            )
            ax.axvspan(
                self.integration_window[0],
                self.integration_window[1],
                0,
                1000*self.ymax,
                color = 'purple',
                alpha = 0.2,
                label = 'integration window',
            )
        # finish figure
        if legend:
            plt.legend()
        if returnfig:
            return fig,axs
        else:
            plt.show()
            pass

    def test_patterns(self,rxs,rys,ymax=None,returnfig=False,legend=True):
        """
        """
        if ymax is not None:
            self.set_ymax(ymax)
        fig,axs = self.select_test_patterns(rxs,rys,returnfig=True,legend=False)
        for idx,(rx,ry) in enumerate(zip(rxs,rys)):
            # get axis
            ax_i = idx//4
            ax_j = idx%4
            ax = axs[ax_i,ax_j]
            fitfun, p0, pcov = self.fit_single_pattern(rx,ry)
            ax.plot(
                self.polarcube.qq,
                fitfun(self.polarcube.qq,*p0),
                color = 'green',
                label = 'background',
            )
            mask_show_bksb = np.logical_and(
                self.polarcube.qq>self.presignal_window[1],
                self.polarcube.qq<=self.postsignal_window[0]
            )
            ax.plot(
                self.polarcube.qq[mask_show_bksb],
                self.median_profile[mask_show_bksb] - fitfun(self.polarcube.qq,*p0)[mask_show_bksb],
                color = 'red',
                label = 'amorphous',
            )
        if legend:
            plt.legend()
        if returnfig:
            return fig,axs
        else:
            plt.show()
            pass

    def fit_amorphous_signal(self):
        ""
        ""
        # allocate space
        im_amorphous = np.zeros(self.datacube.Rshape)
        # loop
        for rx,ry in tqdmnd(self.datacube.Rshape[0],self.datacube.Rshape[1]):
            # Get median profiles
            self._get_median_profiles(rx,ry)
            # Get fit points
            fitfun, p0, pcov = self.fit_single_pattern(rx,ry)
            # populate answer
            im_amorphous[rx,ry] = np.sum(self.median_profile[self.mask_integration] - fitfun(self.polarcube.qq,*p0)[self.mask_integration])
        return im_amorphous


