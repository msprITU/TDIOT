import numpy as np
import cv2
from .utils import cos_window,gaussian2d_rolled_labels
from .fft_tools import fft2,ifft2

class KCF():
#class KCF(BaseCF):
    def __init__(self, padding=1.5, features='gray',interp_factor=0.001):
        super(KCF).__init__()
        self.padding = padding
        self.lambda_ = 1e-4
        self.features = features
        self.w2c=None
        kernel='gaussian'
        if self.features=='gray' or self.features=='color':
            self.interp_factor=interp_factor
            self.sigma=0.2
            self.cell_size=1
            self.output_sigma_factor=0.1
        else:
            raise NotImplementedError
        
        self.kernel=kernel


    def init(self,first_frame,bbox):
        assert len(first_frame.shape)==3 and first_frame.shape[2]==3
        if self.features=='gray':
            first_frame=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self.crop_size = (int(np.floor(w * (1 + self.padding))), int(np.floor(h * (1 + self.padding))))# for vis
        self._center = (np.floor(x0 + w / 2),np.floor(y0 + h / 2))
        self.w, self.h = w, h
        self.window_size=(int(np.floor(w*(1+self.padding)))//self.cell_size,int(np.floor(h*(1+self.padding)))//self.cell_size)
        self._window = cos_window(self.window_size)

        s=np.sqrt(w*h)*self.output_sigma_factor/self.cell_size
        self.yf = fft2(gaussian2d_rolled_labels(self.window_size, s))

        if self.features=='gray' or self.features=='color':
            first_frame = first_frame.astype(np.float32) / 255
            x=self._crop(first_frame,self._center,(w,h))
            x=x-np.mean(x)
        else:
            raise NotImplementedError

        self.xf = fft2(self._get_windowed(x, self._window))
        self.init_response_center = (0,0)
        self.alphaf = self._training(self.xf,self.yf)
        
    def reponse_particle_scale(self,current_frame,candidate_particle_):
        responses_p_arr_max,responses_p_arr_min = [],[]
        response_particle_bbox = []
        response_p_arr_pvpr, response_p_arr_psr, response_p_arr_epsr = [],[],[]
        for i in range(len(candidate_particle_)):
 
            x0_p, y0_p, w_p, h_p = tuple(candidate_particle_[i,:])
            
            if w_p <= 0 or h_p <= 0 or x0_p <0 or y0_p<0:
                
                part_particle_response = [round(x0_p),round(y0_p), w_p,h_p]
                response_particle_bbox.append(part_particle_response)
                responses_p_arr_max.append(0)
                responses_p_arr_min.append(0)
                response_p_arr_pvpr.append(0)
                response_p_arr_psr.append(0)
                response_p_arr_epsr.append(0)
                continue

            if self.features=='gray' or self.features=='color':
                frame_p = current_frame.astype(np.float32) / 255
                self._center_particle = (np.floor(x0_p + (w_p/2)),np.floor(y0_p + (h_p/2)))
                x_p=self._crop(frame_p,self._center_particle,(w_p,h_p))
                
                x_p=x_p-np.mean(x_p)

                fx = self.w/w_p
                fy = self.h/h_p
                x_p_scaled = cv2.resize(x_p,(0,0), fx=fx, fy=fy, interpolation= cv2.INTER_LINEAR)
                zf_p = fft2(self._get_windowed(x_p_scaled, self._window))
                responses_p = self._detection(self.alphaf, self.xf, zf_p, kernel=self.kernel)
                
                curr_p =np.unravel_index(np.argmax(responses_p, axis=None),responses_p.shape)

                responses_p_arr_max.append(np.max(responses_p))
                responses_p_arr_min.append(np.min(responses_p))

                sorted_resp =np.sort(responses_p.flatten())
                max_repsonse_1 = sorted_resp[-1]
                max_repsonse_2 = sorted_resp[-2]
                mean_response = np.mean(responses_p.flatten())
                std_response = np.std(responses_p.flatten())
                
                pvpr = (max_repsonse_1-max_repsonse_2)/max_repsonse_1
                psr = (max_repsonse_1-mean_response)/std_response
                epsr = pvpr*psr
                
                response_p_arr_pvpr.append(pvpr)
                response_p_arr_psr.append(psr)
                response_p_arr_epsr.append(epsr)
                
                if curr_p[0]+1>self.window_size[1]/2:
                    dy_p=fy*(curr_p[0]-self.window_size[1])
                else:
                    dy_p=fy*curr_p[0]
                
                if curr_p[1]+1>self.window_size[0]/2:
                    dx_p=fx*(curr_p[1]-self.window_size[0])
                else:
                    dx_p=fx*curr_p[1]

                dy_p,dx_p=dy_p*self.cell_size,dx_p*self.cell_size
                x_c_p, y_c_p = self._center_particle
                x_c_p+= dx_p
                y_c_p+= dy_p
#                print("dx dy: ", dx_p,dy_p)
#                print("max and min: ",np.max(responses_p),np.min(responses_p))
                
                part_particle_response = [round(x_c_p - w_p / 2),round(y_c_p - h_p/ 2), w_p,h_p]
                response_particle_bbox.append(part_particle_response)

#                img = self._crop(current_frame,self._center_particle,(w_p,h_p))
#                cv2.imshow('image',img)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()

        return np.array(responses_p_arr_max),np.array(responses_p_arr_min), np.array(response_particle_bbox), np.array(response_p_arr_pvpr),np.array(response_p_arr_psr),np.array(response_p_arr_epsr)

    def reponse_particle_no_scale(self,current_frame,candidate_particle_):
        responses_p_arr_max,responses_p_arr_min = [],[]
        response_particle_bbox = []
        response_p_arr_pvpr, response_p_arr_psr, response_p_arr_epsr = [],[],[]
        for i in range(len(candidate_particle_)):
 
            x0_p, y0_p, w_p, h_p = tuple(candidate_particle_[i,:])
            
            if w_p <= 0 or h_p <= 0 or x0_p <0 or y0_p<0:
                
                part_particle_response = [round(x0_p),round(y0_p), w_p,h_p]
                response_particle_bbox.append(part_particle_response)
                responses_p_arr_max.append(0)
                responses_p_arr_min.append(0)
                response_p_arr_pvpr.append(0)
                response_p_arr_psr.append(0)
                response_p_arr_epsr.append(0)
                continue

            if self.features=='gray' or self.features=='color':
                frame_p = current_frame.astype(np.float32) / 255
                self._center_particle = (np.floor(x0_p + (w_p/2)),np.floor(y0_p + (h_p/2)))
                x_p=self._crop(frame_p,self._center_particle,(self.w,self.h))
                x_p=x_p-np.mean(x_p)
                
                zf_p = fft2(self._get_windowed(x_p, self._window))
                responses_p = self._detection(self.alphaf, self.xf, zf_p, kernel=self.kernel)
                
                curr_p =np.unravel_index(np.argmax(responses_p, axis=None),responses_p.shape)

                responses_p_arr_max.append(np.max(responses_p))
                responses_p_arr_min.append(np.min(responses_p))
                
                sorted_resp =np.sort(responses_p.flatten())
                max_repsonse_1 = sorted_resp[-1]
                max_repsonse_2 = sorted_resp[-2]
                mean_response = np.mean(responses_p.flatten())
                std_response = np.std(responses_p.flatten())
                
                pvpr = (max_repsonse_1-max_repsonse_2)/max_repsonse_1
                psr = (max_repsonse_1-mean_response)/std_response
                epsr = pvpr*psr
                
                response_p_arr_pvpr.append(pvpr)
                response_p_arr_psr.append(psr)
                response_p_arr_epsr.append(epsr)
                
                if curr_p[0]+1>self.window_size[1]/2:
                    dy_p=(curr_p[0]-self.window_size[1])
                else:
                    dy_p=curr_p[0]
                
                if curr_p[1]+1>self.window_size[0]/2:
                    dx_p=(curr_p[1]-self.window_size[0])
                else:
                    dx_p=curr_p[1]

                dy_p,dx_p=dy_p*self.cell_size,dx_p*self.cell_size
                x_c_p, y_c_p = self._center_particle
                x_c_p+= dx_p
                y_c_p+= dy_p
#                print("dx dy: ", dx_p,dy_p)
#                print("max and min: ",np.max(responses_p),np.min(responses_p))
                part_particle_response = [round(x_c_p - w_p / 2),round(y_c_p - h_p/ 2), w_p,h_p]
                response_particle_bbox.append(part_particle_response)
                
#                img = self._crop(current_frame,self._center_particle,(self.w,self.h))
#                cv2.imshow('image',img)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()


        return np.array(responses_p_arr_max),np.array(responses_p_arr_min), np.array(response_particle_bbox),np.array(response_p_arr_pvpr),np.array(response_p_arr_psr),np.array(response_p_arr_epsr)


    def get_weight_particle_max_min_response_L2(self,particle_max_response,particle_min_response):
        difference = particle_max_response-particle_min_response
        norm_2 = np.linalg.norm(difference, 2)
        weights = particle_max_response/norm_2
        """
        in order to sum up to one we square it
        """
        weights = weights**2  
        return weights


    def get_weight_particle_max_response_L2(self,particle_max_response):
        norm_2 = np.linalg.norm(particle_max_response, 2)
        weights = particle_max_response/norm_2
        """
        in order to sum up to one we square it
        """
        weights = weights**2   
        return weights
    
        """
        In the two following methods we are doing L1 normalization !!!
        """
    def get_weight_particle_max_response_L1(self,particle_max_response):
        sum_response = np.sum(particle_max_response)
        weights = particle_max_response/sum_response
        return weights
    
    def get_weight_particle_max_min_response_L1(self,particle_max_response,particle_min_response):
        difference = particle_max_response-particle_min_response
        sum_diff = np.sum(difference)
        weights = difference/sum_diff
        return weights
    
    def get_weighted_particle_bbox(self,particles_bbox, weights):
#        w_bbox = np.zeros(4).reshape(1,4)
        w_bbox = np.zeros(4)
        for i in range(len(weights)):
            w_bbox += weights[i]*particles_bbox[i,:]
        return np.rint(w_bbox)

    def eliminate_weak_responses(self, particles, responses, percentage):
        max_response = np.max(responses)
        thresh = (1 - percentage)*max_response
        
        strong_particles = particles[responses >thresh]
        strong_responses = responses[responses >thresh]
        
        return strong_particles, np.array(strong_responses)
        
    def compute_correlation(self,current_frame,candidate_particle_):
        if len(candidate_particle_.shape)==1:
            candidate_particle_=candidate_particle_[:,np.newaxis].T
            
        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        self.responses_p_arr_max,self.responses_p_arr_min, self.response_particle_bbox, self.response_p_arr_pvpr, self.response_p_arr_psr, self.response_p_arr_epsr = self.reponse_particle_scale(current_frame,candidate_particle_)



    def update(self,current_frame):

        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.features=='color' or self.features=='gray':
            current_frame = current_frame.astype(np.float32) / 255
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z=z-np.mean(z)

        zf = fft2(self._get_windowed(z, self._window))
        responses = self._detection(self.alphaf, self.xf, zf, kernel=self.kernel)
        
        curr =np.unravel_index(np.argmax(responses, axis=None),responses.shape)

        if curr[0]+1>self.window_size[1]/2:
            dy=curr[0]-self.window_size[1]
        else:
            dy=curr[0]
        if curr[1]+1>self.window_size[0]/2:
            dx=curr[1]-self.window_size[0]
        else:
            dx=curr[1]
        dy,dx=dy*self.cell_size,dx*self.cell_size
        x_c, y_c = self._center
        x_c+= dx
        y_c+= dy
        self._center = (np.floor(x_c), np.floor(y_c))

        if self.features=='color' or self.features=='gray':
            new_x = self._crop(current_frame, self._center, (self.w, self.h))

        new_xf = fft2(self._get_windowed(new_x, self._window))
        self.alphaf = self.interp_factor * self._training(new_xf, self.yf, kernel=self.kernel) + (1 - self.interp_factor) * self.alphaf
        self.xf = self.interp_factor * new_xf + (1 - self.interp_factor) * self.xf
        
        candidate_bbox= np.array([[(self._center[0] - self.w / 2), (self._center[1] - self.h / 2), self.w, self.h]]).reshape(1,4)
        self.responses_p_arr_max,self.responses_p_arr_min, self.response_particle_bbox, self.response_p_arr_pvpr, self.response_p_arr_psr, self.response_p_arr_epsr = self.reponse_particle_scale(current_frame,candidate_bbox)

        return [(self._center[0] - self.w / 2), (self._center[1] - self.h / 2), self.w, self.h]


    def _kernel_correlation(self, xf, yf, kernel='gaussian'):
        if kernel== 'gaussian':
            N=xf.shape[0]*xf.shape[1]
            xx=(np.dot(xf.flatten().conj().T,xf.flatten())/N)
            yy=(np.dot(yf.flatten().conj().T,yf.flatten())/N)
            xyf=xf*np.conj(yf)
            xy=np.sum(np.real(ifft2(xyf)),axis=2)
            kf = fft2(np.exp(-1 / self.sigma ** 2 * np.clip(xx+yy-2*xy,a_min=0,a_max=None) / np.size(xf)))
        elif kernel== 'linear':
            kf= np.sum(xf*np.conj(yf),axis=2)/np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def _training(self, xf, yf, kernel='gaussian'):
        kf = self._kernel_correlation(xf, xf, kernel)
        alphaf = yf/(kf+self.lambda_)
        return alphaf

    def _detection(self, alphaf, xf, zf, kernel='gaussian'):
        kzf = self._kernel_correlation(zf, xf, kernel)

        responses = np.real(ifft2(alphaf * kzf))
        return responses

    def _crop(self,img,center,target_sz):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        w,h=target_sz
        """
        # the same as matlab code 
        w=int(np.floor((1+self.padding)*w))
        h=int(np.floor((1+self.padding)*h))
        xs=(np.floor(center[0])+np.arange(w)-np.floor(w/2)).astype(np.int64)
        ys=(np.floor(center[1])+np.arange(h)-np.floor(h/2)).astype(np.int64)
        xs[xs<0]=0
        ys[ys<0]=0
        xs[xs>=img.shape[1]]=img.shape[1]-1
        ys[ys>=img.shape[0]]=img.shape[0]-1
        cropped=img[ys,:][:,xs]
        """
        cropped=cv2.getRectSubPix(img,(int(np.floor((1+self.padding)*w)),int(np.floor((1+self.padding)*h))),center)
        return cropped

    def _get_windowed(self,img,cos_window):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        windowed = cos_window[:,:,None] * img
        return windowed