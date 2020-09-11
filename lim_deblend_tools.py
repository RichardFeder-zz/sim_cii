from matplotlib import cm
from scipy.signal import find_peaks

from sparse_tools_F17 import *
from survey_params import *


def n_lc_from_N_lc(N_lc):        
	if type(N_lc)==list:
        n_lc = len(N_lc)
    else:
        n_lc = N_lc

    return n_lc 


def plot_indiv_cov_corr_data_mats(covariance_mat, corr_mat, data_matrix):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(covariance_mat)
    plt.colorbar()
    plt.subplot(1,2,2)

    plt.figure()
    plt.imshow(corr_mat)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title('data matrix')
    plt.imshow(data_matrix.transpose())
    plt.colorbar()
    plt.show()


def Ipred_targ_covariance(Ipred_targ_arr, line_name_array, nsigth=5, return_fig=True):
    
    list_of_covariance_mats = []
    list_of_corr_mats = []
    
    for ilc in range(Ipred_targ_arr.shape[2]):


        data_matrix = np.zeros((Ipred_targ_arr.shape[0], Ipred_targ_arr.shape[1]))

        for n in range(Ipred_targ_arr.shape[0]):
            ravel_nus = []
            for line_idx in range(Ipred_targ_arr.shape[1]):
                ravel_nus.append(np.sum(Ipred_targ_arr[n, line_idx, ilc,:]))
            data_matrix[n,:] = ravel_nus


        covariance_mat = np.cov(data_matrix.transpose())
        corr_mat = np.corrcoef(data_matrix.transpose())

        if np.sum(np.abs(covariance_mat))>0.1:
#             plot_indiv_cov_corr_data_mats(covariance_mat, corr_mat, data_matrix)


            list_of_covariance_mats.append(covariance_mat)
            list_of_corr_mats.append(corr_mat)

                

    mean_covariance = np.nanmean(np.array(list_of_covariance_mats), axis=0)
    mean_corr = np.nanmean(np.array(list_of_corr_mats), axis=0)
    
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle(str(nsigth)+'$\\sigma$ line threshold, $i,j\\in \\lbrace H\\alpha,[OIII],H\\beta,[OII],Ly\\alpha\\rbrace$', fontsize=20, y=1.02)
    plt.subplot(1,2,1)
    plt.title('$\\frac{1}{N_{lc}}\\sum_{n=1}^{N_{lc}}\\mathcal{C}(I_i,I_j)$', fontsize=16)
    
    plt.imshow(mean_covariance, vmin=np.percentile(mean_covariance, 15), vmax=np.percentile(mean_covariance, 95), origin='lower')
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('[Jy$^2$/sr$^2$]', fontsize=16)
    plt.xticks(np.arange(5), line_name_array, fontsize=13)
    plt.yticks(np.arange(5), line_name_array, fontsize=13)
    plt.subplot(1,2,2)
    plt.title('$\\frac{1}{N_{lc}}\\sum_{n=1}^{N_{lc}}\\rho(I_i,I_j)$', fontsize=16)
    plt.imshow(mean_corr, vmin=-1, vmax=1, origin='lower')
    plt.colorbar(orientation='horizontal')
    plt.xticks(np.arange(5), line_name_array, fontsize=13)
    plt.yticks(np.arange(5), line_name_array, fontsize=13)

    plt.tight_layout()
    plt.show()
    
    if return_fig:
        return fig, mean_covariance, mean_corr, list_of_covariance_mats, list_of_corr_mats
    else:
        return mean_covariance, mean_corr, list_of_covariance_mats, list_of_corr_mats

class sphx_test():
    
    model = 'Be13'
    
    def __init__(self, nsig_th=0.97):
        self.sphx_par = spherex_param()
        self.nsig_th = nsig_th
        
        self.dOm = (self.sphx_par.dth * u.arcmin.to(u.rad))**2
        
        self.line_use = ['Ha', 'OIII', 'Hb', 'OII', 'Lya']
        self.line_targ_vec = self.line_use.copy()

        self.line_name_arr = [r'$H\alpha$', r'$[O\, III]$', r'$H\beta$', r'$[O\, II]$', r'$Ly\alpha$']
        self.nu0_arr = [spec_lines.Ha.to(u.GHz, equivalencies=u.spectral()).value,\
           spec_lines.OIII.to(u.GHz, equivalencies=u.spectral()).value,\
           spec_lines.Hb.to(u.GHz, equivalencies=u.spectral()).value,\
           spec_lines.OII.to(u.GHz, equivalencies=u.spectral()).value,\
           spec_lines.Lya.to(u.GHz, equivalencies=u.spectral()).value]
        self.sigI0 = 3631 * 10**(-22 / 2.5) / 5 / self.dOm / np.sqrt(4) # pixel NEI
        self.sigI = self.sigI0 / np.sqrt(4)
        self.e_th = self.sigI * self.nsig_th # normally * 0.97

        
        
    def load_dictionary(self, filename='data_internal/sparse_be13_dict.pickle'):
        
        with open(filename,'rb') as pfile:
            self.A, self.I_norm, self.z_coords, self.N_nu, self.N_z, \
                        self.z_coords_all, self.z_idx, self.I_coords_all = pickle.load(pfile)
                
    def load_observations(self, filename='data_internal/sparse_be13_general_Lr.pickle'):
        with open(filename,'rb') as pfile:
            self.N_true, self.Itrue_all, self.Itrue_targ = pickle.load(pfile)
            
        self.noise = np.random.normal(0, self.sigI, self.Itrue_all.shape)

          
    def sim_noise(self, nsims=10, N_lc=None, iter_max= 10, verbose=False, show_iter_plots=False, stochastic=False, \
                 fast_lines=False, amp_nsigma=5.):
        
        ''' this function allows two types of marginalization. The first considers the results of MP on an ensemble of data realizations including noise,
        while the second runs a stochastic version of MP producing several model realizations based on a single data realization. '''

        if N_lc is None:
            N_lc = self.Itrue_all.shape[0]
            
        n_lc = n_lc_from_N_lc(N_lc)
            
        f_arrs = []
        iter_counts = []
        Ipred_targ_arr = np.zeros([nsims, len(self.line_use), n_lc, self.Itrue_all.shape[-1]])
        print('Ipred targ has shape ', Ipred_targ_arr.shape)

        if stochastic:
            Iobs_all = self.Itrue_all + self.noise
            
            if type(N_lc)==list:
                Iobs_specific_lines = Iobs_all[np.array(N_lc)]
            else:
                Iobs_specific_lines = Iobs_all[:N_lc]
            
            for i in range(nsims):
                npred, ipredtarg, idx_order, f_arr, iter_cts = self.run_MP_ampth(Iobs_all, N_lc=N_lc, iter_max=iter_max, return_Ipred_targ=True,\
                                            verbose=verbose, show_iter_plots=show_iter_plots, realization_noise=self.noise, \
                                           stochastic=stochastic, fast_lines=fast_lines, amp_nsigma=amp_nsigma)
            
                Ipred_targ_arr[i] = ipredtarg
                
                f_arrs.append(f_arr)
                iter_counts.append(iter_cts)
            
                if n_lc > 100:
                    print('i = ', i)
            
        
        else:
            for i in range(nsims):
                if nsims > 1:
                    noise = np.random.normal(0, self.sigI, self.Itrue_all.shape)
                    Iobs_all = self.Itrue_all + noise
                else:
                    Iobs_all = self.Itrue_all + self.noise
                    
                if type(N_lc)==list:
                    Iobs_specific_lines = Iobs_all[np.array(N_lc)]
                else:
                    Iobs_specific_lines = Iobs_all[:N_lc]

                    
                npred, ipredtarg, idx_order, f_arr, iter_cts = self.run_MP_ampth(Iobs_all, N_lc=N_lc, iter_max=iter_max, return_Ipred_targ=True,\
                                                verbose=verbose, show_iter_plots=show_iter_plots, realization_noise=self.noise, \
                                               stochastic=stochastic, fast_lines=fast_lines, amp_nsigma=amp_nsigma)

                Ipred_targ_arr[i] = ipredtarg
                
                f_arrs.append(f_arr)
                iter_counts.append(iter_cts)

            
        return Ipred_targ_arr, f_arrs, iter_counts, Iobs_specific_lines
    
    def run_MP_ampth(self, Iobs_all, N_lc=None, A=None, I_norm=None, I_norm_divfac=4, e_th=None, iter_max = 10, stochastic=False,\
                verbose=False, return_Ipred_targ=False, show_iter_plots=False, realization_noise=None, fast_lines=True, \
               amp_nsigma=5., nstrikes=3):

        
        if N_lc is None:
            N_lc = Iobs_all.shape[0] # number of light curves

		n_lc = n_lc_from_N_lc(N_lc)

        if e_th is None:
            e_th = self.e_th
            
        idx_order, iter_cts, f_arr_lines = [], [], []

        N_nu, N_z = self.A.shape # number of frequency channels, number of redshifts?
        N_pred = np.zeros([n_lc, N_z]) # predicted N, number of light cones x number of redshift bins

        if verbose:
            print('N_nu, N_z:', N_nu, N_z)
            print('N_lc:', n_lc)
                        
        if fast_lines: # NOT ACTIVELY UPDATED AT THE MOMENT, but worked at one point. This procedure scales better to larger numbers of lines
            
            t0 = time.time()
            eth_bool_array = np.zeros((n_lc)) # threshold boolean array for each sight line
            iter_count = 0 # number of iterations
            all_R_arrs = Iobs_all[:N_lc].copy() # copy the number of lines we're dealing with 
            all_f_arrs = np.zeros_like(all_R_arrs) # this will hold the reconstructions
            
            all_NI_arr = np.zeros([N_lc, N_z]) # this has effective number counts
            
            # run while loop until all light curves have reached threshold or when iter counter reaches max

            while np.sum(eth_bool_array) < Iobs_all.shape[0] and iter_count < iter_max:
                # the big dot product is between the simulated data matrix (N_sightlines x N_nu) and the template
                # (N_templates x N_nu) transpose A^T , so we end up with (N_sightlines x N_templates) which contains the 
                # inner products of the templates with each sightline
                iter_count += 1
                big_dotprod = np.dot(all_R_arrs, self.A)

                # choose the argmax for each sightline 
                all_gammas = np.argmax(big_dotprod, axis=1) # indices of the chosen templates for each sightline

                # compute amplitudes of the templates 
                all_amp = np.array([np.sum(self.A[:,all_gammas[i]] * all_R_arrs[i]) for i in range(N_lc)])
                
                # all_u should be the added template for all sightlines, so N_lc x N_nu
                all_u = np.array([all_amp[i] * self.A[:,all_gammas[i]] for i in range(N_lc)])

                all_f_arrs[~eth_bool_array.astype(np.bool)] += all_u[~eth_bool_array.astype(np.bool)]
                all_R_arrs[~eth_bool_array.astype(np.bool)] -= all_u[~eth_bool_array.astype(np.bool)]

                # compute RMS and determine which are below threshold
                all_R = np.array([np.sqrt(np.mean(R_arr**2, axis=0)) for R_arr in all_R_arrs])

                eth_bool_array[all_R < e_th] = 1
            
                t1 = time.time()
            
                for i in range(N_lc):
                    if not eth_bool_array[i]:
                        all_NI_arr[i,all_gammas[i]] += all_amp[i]
                
                print('time for dumb loop is ', time.time()-t1)

                
            print('out of the while loop')
            N_pred = all_NI_arr / (self.I_norm / I_norm_divfac)
            print('time elapsed for fast version is ', time.time()-t0)

            if return_Ipred_targ:
                _, Ipred_targ = gen_Ipred(self.z_coords, N_pred, self.sphx_par.dth*2, self.sphx_par.nu_binedges,\
                                       self.line_use, self.line_targ_vec, model = self.model, verbose = verbose)

                print(N_pred.shape, Ipred_targ.shape, idx_order, all_f_arrs.shape, iter_cts)
                return N_pred, Ipred_targ, idx_order, all_f_arrs, iter_cts
                
        else:
                
            t0 = time.time()

            if type(N_lc)==list:
                iter_range = N_lc
            else:
                iter_range = np.arange(N_lc)
                
            for i, ilc in enumerate(iter_range):
                

                idx_ord = []
                R_arr = Iobs_all[ilc].copy()

                if verbose:
                    print('R_arr:', R_arr)
                R = np.sqrt(np.mean(R_arr**2)) # the initial residual is the RMS of the component
                f_arr, NI_arr = np.zeros(N_nu), np.zeros(N_z)

                iter_count = 0
                ntry = 0

                # start the MP algorithm
                while True:
                    if verbose:
                        print('top of while loop')
                    # if iteration counter reaches max, end it
                    if iter_count == iter_max:
                    	if verbose:
                    		print('reached maximum of iteration counter, breaking from while loop')
                        break
                        
                    dotprod = np.dot(R_arr.reshape(1, -1), self.A)[0]

                    # this chooses the top 10 templates, but should just choose all the templates above the threshold (TODO)
                    nsort = 10
                    ind = np.argpartition(dotprod, -nsort)[-nsort:]

                    # choose the dictionary element with the largest inner product
                    if stochastic:

                        probs = dotprod[ind] / np.sum(dotprod[ind])
                        gamma = np.random.choice(ind, p=probs)
                        amp = np.sum(self.A[:,gamma] * R_arr)
                        
                        if amp < amp_nsigma*self.sigI:
                            if ntry < nstrikes:
                                ntry += 1
                                if verbose:
                                    print('ntry is now ', ntry, ' going to top of while loop, ', amp, amp_nsigma*self.sigI)
                                continue       
                            else: # if you run out of tries, then just look at largest dot product and check that
                                if verbose:
                                    print('ran out of strikes, trying largest dot product')
                                gamma = np.argmax(dotprod)
                                amp = np.sum(self.A[:,gamma] * R_arr)
                                # if amplitude is still smaller than threshold, then we're done. otherwise, we choose it
                                if amp < amp_nsigma*self.sigI:
                                    if verbose:
                                        print('largest dot product still smaller, breaking ', amp, amp_nsigma*self.sigI)
                                    break
                                    
                        # once we find one that works, add it to the idx_order array and reset the stochastic counter
                        if verbose:
                            print('found one that works, ', gamma, amp, amp_nsigma*self.sigI)
                        idx_ord.append(gamma)
                        ntry = 0
                    
                    else:
                        gamma = np.argmax(dotprod) # this is the index of the template
                        idx_ord.append(gamma)

                        amp = np.sum(self.A[:,gamma] * R_arr)
                        
                        if amp < amp_nsigma*self.sigI:
                            break

                    iter_count += 1

                    # multiply the amplitude by the normalized template
                    u = amp * self.A[:,gamma]

                    if show_iter_plots:
                        self.plot_model_iter(R_arr, u, e_th, ilc, iter_count, gamma, realization_noise=realization_noise)
                    
                    NI_arr[gamma] += amp
                    # subtract largest inner product from residual
                    R_arr -= u
                    f_arr += u
                    R = np.sqrt(np.mean(R_arr**2))
                
                N_pred[i,:] = NI_arr / (self.I_norm / I_norm_divfac)

                # N_pred[ilc,:] = NI_arr / (self.I_norm / I_norm_divfac)
                iter_cts.append(iter_count)
                idx_order.append(idx_ord)
                f_arr_lines.append(f_arr)
            
            print('f_arr_lines has shape ', np.array(f_arr_lines).shape)
        
            if verbose:
                print('time elapsed for slow version is ', time.time()-t0)
                print('N_pred has shape ', N_pred.shape)
                print(N_pred)

            if return_Ipred_targ:
                _, Ipred_targ = gen_Ipred(self.z_coords, N_pred, self.sphx_par.dth*2, self.sphx_par.nu_binedges,\
                                       self.line_use, self.line_targ_vec, model = self.model, verbose = verbose)

                return N_pred, Ipred_targ, idx_order, f_arr_lines, iter_cts

        return N_pred
    
    def plot_model_iter(self, R_arr, u, e_th, ilc, iter_count, gamma, realization_noise=None):
        
        plt.figure()
        plt.title('Iteration '+str(iter_count)+': Template index = '+str(gamma)+' z='+str(np.round(self.z_coords[gamma], 2)))
        plt.plot(np.arange(len(R_arr)), R_arr, color='k', linewidth=2, label='current residual')
        plt.plot(np.arange(len(u)), u, color='r', linewidth=1, label='chosen template')
        if realization_noise is not None:
            plt.plot(np.arange(len(u)), realization_noise[ilc], color='b', linestyle='dashed', label='noise')
        plt.axhline(-e_th, color='k', linewidth=5)
        plt.axhline(e_th, color='k', linewidth=5)
        plt.xlabel('Frequency bin index')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
    
    def show_result1(self, Ipred_targ_arr, N_lc=None, stochastic=False):
        
        if N_lc is None:
            N_lc = Ipred_targ_arr.shape[2]

        fig, ax = plt.subplots(5, 1, figsize = (10,25))

        # for each spectral line/array...
        for jtarg, (nu0, line_name) in enumerate(zip(self.nu0_arr, self.line_name_arr)):
            
            # redshift bins obtained by rest frame divided by observed frequency - 1
            zbins = nu0 / self.sphx_par.nu_bins - 1
            
            # we know what the true mean intensity is for the light cones we choose
            mu_true = np.mean(self.Itrue_targ[jtarg,:N_lc,:], axis=0)
            ax[jtarg].plot(zbins, mu_true, 'ko-', markersize = 5, lw = 2)
            
            # compute mean across light cones
            mus_pred = np.mean(Ipred_targ_arr[:,jtarg,:N_lc,:], axis = 1)
            # then take mean and standard deviation across realizations
            mu_pred = np.mean(mus_pred, axis = 0)
            sig = np.std(mus_pred, axis = 0)
            # plot them!
            ax[jtarg].errorbar(zbins, mu_pred, sig, c = 'r', fmt = 'none', capsize = 2, lw = 3)
            if stochastic:
                ax[jtarg].set_title(line_name+' (stochastic MP)', fontsize = 20)
            else:
                ax[jtarg].set_title(line_name, fontsize = 20)

            ax[jtarg].tick_params(axis='both', which='major', labelsize=15)
            ax[jtarg].set_ylabel('I [Jy/sr]', fontsize = 20)
        ax[4].set_xlabel('z', fontsize = 20)
        fig.tight_layout()
        return fig
        
        
    def plot_correlation_coeffs_sphx(self, Ipred_targ_arr, N_lc=None, return_fig=False):
    
        if N_lc is None:
            N_lc = Ipred_targ_arr.shape[2]
        fig, ax = plt.subplots(1, 1, figsize = (15,7))

        mucntot_arr = np.array([])
        sigcntot_arr = np.array([])
        ztot_arr = np.array([])
        
        for jtarg in range(len(x.line_use)):
            nu0 = self.nu0_arr[jtarg]
            zbins = nu0 / sphx_par.nu_bins - 1
            sp = np.where(zbins < 10)[0]
            mucp_arr, sigcp_arr, mucn_arr, sigcn_arr = [np.zeros(self.N_nu) for x in range(4)]
            for iband in range(self.N_nu):
                cp_arr, cn_arr = [np.zeros(Ipred_targ_arr.shape[0]) for x in range(2)]
                for isim in range(Ipred_targ_arr.shape[0]):
                    mapt = self.Itrue_targ[jtarg, :N_lc, iband]
                    mapp = Ipred_targ_arr[isim, jtarg, :N_lc, iband].copy()
                    mapn = np.random.normal(size = mapt.shape)
                    mapt -= np.mean(mapt)
                    mapp -= np.mean(mapp)
                    mapn -= np.mean(mapn)
                    if np.sum(mapt**2) > 0 and np.sum(mapp**2) > 0:
                        cp_arr[isim] = np.sum(mapt * mapp) / np.sqrt(np.sum(mapt**2) * np.sum(mapp**2))
                    if np.sum(mapt**2) > 0 and np.sum(mapn**2) > 0:
                        cn_arr[isim] = np.sum(mapt * mapn) / np.sqrt(np.sum(mapt**2) * np.sum(mapn**2))
                mucp_arr[iband] = np.mean(cp_arr)
                sigcp_arr[iband] = np.std(cp_arr)
                mucn_arr[iband] = np.mean(cn_arr)
                sigcn_arr[iband] = np.std(cn_arr)

            mucntot_arr = np.concatenate((mucntot_arr, mucn_arr[sp]))
            sigcntot_arr = np.concatenate((sigcntot_arr, sigcn_arr[sp]))
            ztot_arr = np.concatenate((ztot_arr, zbins[sp]))
            ax.fill_between(zbins[sp],mucp_arr[sp] - sigcp_arr[sp], mucp_arr[sp] + sigcp_arr[sp],\
                            label = line_name_arr[jtarg], alpha = 0.6)
        sortidx = ztot_arr.argsort()
        ax.fill_between(ztot_arr[sortidx], mucntot_arr[sortidx] - sigcntot_arr[sortidx],\
                        mucntot_arr[sortidx] + sigcntot_arr[sortidx], alpha=0.6, facecolor='gray')
        ax.legend(loc = 0, fontsize = 20)
        ax.set_xlabel('z', fontsize = 20)
        ax.set_xlim([0,10])
        ax.set_ylabel(r'correlation coeff. r', fontsize = 20)
        ax.set_xticks(np.arange(10))
        ax.tick_params(axis='both', which='major', labelsize=15)

        if return_fig:
            return fig


        
    def Ipred_targ_redshift_covariance(self, Ipred_targ_arr, line_name_array, f_arr_pred=None, ilc_idx=None, nsigth=5, return_fig=True, zoom=True):
    
        sum_covariances, list_of_template_idxs = [], []

        fig = plt.figure(figsize=(7, 8))
#         plt.suptitle('$\\sum_{n=1}^{N_{lc}}\\mathcal{C}_n(I_{z_i},I_{z_j})$, '+str(nsigth)+'$\\sigma$ stopping threshold', fontsize=20)
        plt.suptitle('$\\rho(I_{z_i},I_{z_j})$, '+str(nsigth)+'$\\sigma$ stopping threshold', fontsize=20, y=1.05)

        
        nothin_counter = False
        for line_idx in range(Ipred_targ_arr.shape[1]-1):

            zbins = self.nu0_arr[line_idx] / self.sphx_par.nu_bins - 1
            list_of_covariance_mats = []
            
            if len(Ipred_targ_arr.shape)==3:
                data_matrix = np.zeros((Ipred_targ_arr.shape[0], Ipred_targ_arr.shape[-1]))
                for n in range(Ipred_targ_arr.shape[0]):
                    data_matrix[n,:] = Ipred_targ_arr[n, line_idx]
   
            else:
                for ilc in range(Ipred_targ_arr.shape[2]):
                    data_matrix = np.zeros((Ipred_targ_arr.shape[0], Ipred_targ_arr.shape[-1]))
                    for n in range(Ipred_targ_arr.shape[0]):
                        data_matrix[n,:] = Ipred_targ_arr[n, line_idx,ilc]

            covariance_mat = np.cov(data_matrix.transpose())


            if np.sum(np.abs(covariance_mat))>0.1:
                list_of_covariance_mats.append(covariance_mat)

            sum_covariance = np.nansum(np.array(list_of_covariance_mats), axis=0)
            sum_covariances.append(sum_covariance)
            plt.subplot(2,2,line_idx+1)
#             plt.subplot(1,Ipred_targ_arr.shape[1]-1,line_idx+1)
            plt.title(line_name_array[line_idx], fontsize=16)
            select_idxs = None
            if len(list_of_covariance_mats) > 0:
                
                print('sum_cov.shape[0]:', sum_covariance.shape[0])
                for j in range(sum_covariance.shape[0]):
                    for n, idxs in enumerate(np.nonzero(sum_covariance[j])):
                        if len(idxs) > 1:
                            select_idxs = idxs
                            nothin_counter = True
                            break
                            
                print('select idxs is ', select_idxs) 
#                 print(np.nonzero(sum_covariance[j])[n])

                if select_idxs is not None:
                    sum_cov_zoom = sum_covariance[select_idxs]
                    sum_cov_zoom = sum_cov_zoom[:, select_idxs]
                
                    sum_corr_zoom = np.array([[sum_cov_zoom[i,j]/np.sqrt(sum_cov_zoom[i,i]*sum_cov_zoom[j,j]) for i in range(sum_cov_zoom.shape[0])] for j in range(sum_cov_zoom.shape[1])])
                
#                 print(sum_cov_zoom)
#                 plt.imshow(sum_cov_zoom, vmin=np.min(sum_cov_zoom), vmax=np.max(sum_cov_zoom), origin='lower')
                    plt.imshow(sum_corr_zoom, vmin=-1, vmax=1, origin='lower', cmap='bwr')
                else:
                    plt.imshow(np.zeros((2,2)), cmap='bwr')
#                 plt.imshow(sum_cov_zoom, vmin=np.percentile(sum_cov_zoom, 15), vmax=np.percentile(sum_cov_zoom, 95), origin='lower')

                
                plt.xlabel('$z$', fontsize=16)
                plt.ylabel('$z$', fontsize=16)
#                 tick_idxs = np.array([0, 20, 40, 60, 80])
#                 mask = (tick_idxs > min_idx)*(tick_idxs < max_idx)
#                 tick_idxs = tick_idxs[mask]
                
                if zoom and select_idxs is not None:
                    plt.xticks(np.arange(len(select_idxs)), np.round(zbins[select_idxs], 2))
                    plt.yticks(np.arange(len(select_idxs)), np.round(zbins[select_idxs], 2))


#                     plt.yticks(tick_idxs, np.round(zbins[tick_idxs], 1))
                
#                 plt.imshow(sum_covariance, vmin=np.percentile(sum_covariance, 15), vmax=np.percentile(sum_covariance, 95), origin='lower')
                cbar = plt.colorbar(orientation='horizontal')
#                     cbar.set_label('[Jy$^2$/sr$^2$]', fontsize=16)
            list_of_template_idxs.append(select_idxs)
#             plt.xlabel('$z$', fontsize=16)
#             plt.ylabel('$z$', fontsize=16)
#             tick_idxs = np.array([0, 20, 40, 60, 80])
#             plt.xticks(tick_idxs, np.round(zbins[tick_idxs], 1))
#             plt.yticks(tick_idxs, np.round(zbins[tick_idxs], 1))

        if not nothin_counter:
            plt.close()
            fig = None
        else:
            plt.tight_layout()
            plt.show()
            
        f = None
        
        if nothin_counter:
            colors = ['C0', 'C1', 'C2', 'C3', 'C4']
            markers = ['o', '*', 'v', 'x', '+', '^', 'v', 'x', '+', '^']
            if ilc_idx is not None:
                f, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 14))
            else:
                f, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
#             plt.figure(figsize=(10, 10))
            for line_idx in range(Ipred_targ_arr.shape[1]-1):
                if list_of_template_idxs[line_idx] is not None:
                    cmap = cm.get_cmap('viridis')
                    redshifts = zbins[np.array(list_of_template_idxs[line_idx])]
                    normredshifts = np.arange(len(redshifts))/len(redshifts)
#                     normredshifts = redshifts/np.max(redshifts)
                    
#                     print(normredshifts)
                    cmap_colors = cmap(normredshifts)
            
                    zbins = self.nu0_arr[line_idx] / self.sphx_par.nu_bins - 1

#                 plt.subplot(4,1,line_idx+1)
#                 plt.title(self.line_name_arr[line_idx], fontsize=16)
                if list_of_template_idxs[line_idx] is not None:
                    all_nonz_A = []
            
                    xtextpos = []
                    ytextpos = []
            
                    for n, tidx in enumerate(list_of_template_idxs[line_idx]):
                        
                        label = 'z = '+str(np.round(zbins[tidx], 2))
#                         plt.plot(np.arange(self.A.shape[0]), self.A[:,self.A.shape[0]*line_idx+tidx],label=label, color=colors[line_idx], alpha=0.2)
                        axes[line_idx].plot(np.arange(self.A.shape[0]), self.A[:,self.A.shape[0]*line_idx+tidx],label=label, color=cmap_colors[n])
                        
                        axes[line_idx].legend(fontsize=10)

                        A_vec =  self.A[:,self.A.shape[0]*line_idx+tidx]
                        nonz_A = np.where(A_vec > 0.)[0]
                        peaks, _ = find_peaks(A_vec, height=0)
                        
#                         all_nonz_A.extend(nonz_A)
#                         plt.scatter(nonz_A, A_vec[nonz_A], marker=markers[n], color=colors[line_idx])
#                         offsets = np.arange(len(nonz_A))-np.mean(np.arange(len(nonz_A)))
#                         print(offsets)
                        for pidx, peak in enumerate(peaks):
                            fac = 0.0
                            if peak in xtextpos:
#                                 print('peak is in xtextpos!')
                                fac = 0.1
                                if A_vec[peak] in ytextpos:
#                                     print('y val is too!')
                                    fac = 0.1
                            
#                             axes[line_idx].text(peak, fac+A_vec[peak], label, fontsize=11)
                            xtextpos.append(peak)
                            ytextpos.append(A_vec[peak])

#                         plt.scatter(nonz_A, A_vec[nonz_A], marker=markers[n], s=200, color=colors[line_idx])
#                 plt.xlim(np.min(all_nonz_A)-5)
#                 if line_idx == Ipred_targ_arr.shape[1]-1:
                axes[line_idx].set_xlabel('$\\nu$ index', fontsize=12)
#                 axes[line_idx].set_ylim(0, 1.1)
                axes[line_idx].text(75, 0.65, self.line_name_arr[line_idx], fontsize=24)
            
            if ilc_idx is not None:
                axes[4].plot(np.arange(self.A.shape[0]), self.Itrue_all[ilc_idx], color='k', label='truth')
                axes[4].plot(np.arange(self.A.shape[0]), np.median(f_arr_pred, axis=0), color='b', label='pred')
                axes[4].fill_between(np.arange(self.A.shape[0]), np.percentile(f_arr_pred, 16, axis=0), np.percentile(f_arr_pred, 84, axis=0), color='b', alpha=0.5)
        
            plt.tight_layout()
            plt.show()
            
        
        if return_fig:
            return fig, f, sum_covariances
        
        return sum_covariances

def compare_greedy_stochastic_mp_individual_line(nuidxs, Itrue_line, Iobs_line, f_arr_greedy, f_arrs_st, noise=None, nsigma_th=3.0, return_fig=True):
    
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(str(nsigma_th)+'$\\sigma$ threshold', fontsize=20, y=1.03)
    
    plt.subplot(3,1,1)
    plt.title('Greedy Matched Pursuit', fontsize=14)
    plt.plot(nuidxs, Itrue_line, color='k', label='truth', linewidth=4, marker='.')
    plt.plot(nuidxs, Iobs_line, label='observed', linewidth=2, color='g', linestyle='dashed')
    if noise is not None:
        plt.plot(nuidxs, noise, color='grey', label='noise')
    plt.plot(nuidxs, f_arr_greedy, label='Greedy recon.', color='r', linewidth=1)
    plt.xlabel('$\\nu$ index', fontsize=14)
    plt.ylabel('I [Jy/sr]', fontsize=14)
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.title('Stochastic Matched Pursuit (Shaded regions bound 16th and 84th pcts.)', fontsize=14)
    plt.plot(nuidxs, Itrue_line, color='k', label='truth', linewidth=4, marker='.')
    plt.plot(nuidxs, Iobs_line, label='observed', linewidth=2, color='g', linestyle='dashed')
    if noise is not None:
        plt.plot(nuidxs, noise, color='grey', label='noise')
    plt.plot(nuidxs, np.median(f_arrs_st, axis=0), color='C0', label='Median stochastic recon.', linewidth=1)
    plt.fill_between(nuidxs, np.percentile(f_arrs_st, 16, axis=0), np.percentile(f_arrs_st, 84, axis=0),color='C0', alpha=0.5)
    plt.legend()
    plt.xlabel('$\\nu$ index', fontsize=14)
    plt.ylabel('I [Jy/sr]', fontsize=14)
    
    plt.subplot(3,1,3)
    plt.title('Residuals', fontsize=14)
    plt.plot(nuidxs, np.median(f_arrs_st, axis=0)-Itrue_line, color='C0', label='Stochastic', linewidth=2, zorder=10)
    plt.fill_between(nuidxs, np.percentile(f_arrs_st, 16, axis=0)-Itrue_line, np.percentile(f_arrs_st, 84, axis=0)-Itrue_line, color='C0', alpha=0.5)
    plt.plot(nuidxs, f_arr_greedy-Itrue_line, color='r', label='Greedy', linewidth=2)
    if noise is not None:
        plt.plot(nuidxs, noise, color='grey', label='noise')
    plt.legend()
    plt.xlabel('$\\nu$ index', fontsize=14)
    plt.ylabel('I [Jy/sr]', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    if return_fig:
        return fig


