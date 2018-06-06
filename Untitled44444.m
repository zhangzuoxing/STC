
%%
clc;close all;clear all;
%%
fftw('planner','patient')
%% set path
%addpath('./micsigo');
%img_dir = dir('./micsigo/*.bmp');
%addpath('./A3000');
%img_dir = dir('./A3000/*.bmp');
addpath('./jiu_0');
img_dir = dir('./jiu_0/*.bmp');
%addpath('./jiu_2');
%img_dir = dir('./jiu_2/*.bmp');
%addpath('./data');
%img_dir = dir('./data/*.jpg');
%addpath('./cut');
%img_dir = dir('./cut/*.bmp');
%addpath('./SR');
%img_dir = dir('./SR/*.bmp');
%addpath('./skybox');
%img_dir = dir('./skybox/*.bmp');
%% initialization
%initstate = [478,864,67,68];%SR
%%% 
%initstate = [196,336,36,36];%skybox
%initstate = [95,135,2,2];%skybox
%initstate = [127,105,8,8];%skybox
%initstate = [552,64,9,9];%skybox
%initstate = [161,65,75,95];%initial rectangle [x,y,width, height]
%initstate = [914,673,26,26];%jiu_2
%initstate = [862,618,21,18];%jiu_2
%initstate = [848,623,18,17];%jiu_2
%initstate = [1268,1362,15,18];%jiu_2
%initstate = [1303,1495,21,21];%jiu_2
%initstate = [1220,1328,21,21];%jiu_2
%initstate = [1162,1127,23,23];%jiu_2
initstate = [1030,904,42,42];%jiu-0
%initstate = [595,326,17,13];%micigo
%initstate = [1048,298,12,12];%micigo\
%initstate = [313,324,13,13];%micigo
%initstate = [937,316,20,16];%micigo
%initstate = [645,325,15,9];%micigo
%initstate = [829,1093,48,62];%initial rectangle [x,y,width, height] airport padding=0.9美利达A3000
%initstate = [840,1108,40,38];%initial rectangle [x,y,width, height] airport 美利达A3000
%initstate = [1337,766,19,18];%initial rectangle [x,y,width, height] CUT
%initstate = [1086,622,40,40];%initial rectangle [x,y,width, height]CUT
%initstate = [562,626,40,40];%initial rectangle [x,y,width, height]CUT
pos = [initstate(2)+initstate(4)/2 initstate(1)+initstate(3)/2];%center of the target
target_sz = [initstate(4) initstate(3)];%initial size of the target
%% parameters according to the paper
padding =0.9;					%extra area surrounding the target
rho = 0.075;			        %the learning parameter \rho in Eq.(12)
sz = floor(target_sz * (1 + padding));% size of context region
%% parameters of scale update. See Eq.(15)
scale = 1;%initial scale ratio
lambda = 0.1;% \lambda in Eq.(15)
num = 5; % number of average frames
%% store pre-computed confidence map
alapha = 1;                    %parmeter \alpha in Eq.(6)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
dist = rs.^2 + cs.^2;
conf = exp(-0.5 / (alapha) * sqrt(dist));%confidence map function Eq.(6)
conf = conf/sum(sum(conf));% normalization
conff = fft2(conf); %transform conf to frequencey domain
%% store pre-computed weight window
hamming_window = hamming(sz(1)) * hann(sz(2))';
sigma = mean(target_sz);%sigma = mean(target_sz);% initial \sigma_1 for the weight function w_{\sigma} in Eq.(11)
window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));% use Hamming window to reduce frequency effect of image boundary
window = window/sum(sum(window));%normalization
%%
b_weight=target_sz(1);%目标框宽
b_height=target_sz(2);%目标框高
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for frame = 1:numel(img_dir)
    sigma = sigma*scale;% update scale in Eq.(15)
    window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));%update weight function w_{sigma} in Eq.(11)
    window = window/sum(sum(window));%normalization
 	%load image
    img = imread(img_dir(frame).name);	    
	if size(img,3) > 1,
		im = rgb2gray(img);
	end
   	contextprior = get_context(im, pos, sz, window);% the context prior model Eq.(4)
    %R=im((pos(1)-b_weight):(pos(1)+b_weight),(pos(2)-b_height):(pos(2)+b_height));
    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%kalman_pos=[X(1),X(2)];
%contextprior = get_context(im, kalman_pos, sz, window);% the context prior model Eq.(4)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if frame > 1,
		%calculate response of the confidence map at all locations
	    confmap = real(ifft2(Hstcf.*fft2(contextprior))); %Eq.(11) 
       	%target location is at the maximum response
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%找到响应峰值最大的几个点的位置%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        t=sort(confmap(:));
        t=t';
        t=(fliplr(t))';
        [m,n]=find(confmap>=t(5),5);
        t=[m,n];
        [row1, col1] = find(confmap == max(confmap(:)), 1);%响应峰值最大位置
        %%%%%%%%%%%%%对比几个峰值和最大值的距离%%%%%%%%%%
        for ii=1:length(t)
            if sqrt((m(ii)-row1)^2+(n(ii)-col1)^2)>=min(b_height,b_weight)
                m(ii)=row1;
                n(ii)=col1;
            end
        end       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%上帧目标框哈希值计算%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        R_x1=floor(pos(1)-0.5*b_weight);
        R_x2=floor(pos(1)+0.5*b_weight);
        R_y1=floor(pos(2)-0.5*b_height);
        R_y2=floor(pos(2)+0.5*b_height);
        if R_x1<=0
            R_x1=1;
        end
        if R_y1<=0
            R_y1=1;
        end
        AA=size(im);
        if R_x2>=AA(1);
            R_x2=AA(1)-1;
        end 
        if R_y2>=AA(2);
            R_y2=AA(2)-1;
        end  
        R=im(R_x1:R_x2,R_y1:R_y2);%
        %R=im(floor(pos(1)-0.5*b_weight):floor(pos(1)+0.5*b_weight),floor(pos(2)-0.5*b_height):floor(pos(2)+0.5*b_height));%   
        if b_weight>=b_height
            nr=b_weight;
        else
            nr=b_height;
        end
        R=imresize(R,[nr,nr]);
        R=dct2(R);
        me_R=mean(mean(R));
        ni=8;
        for i=1:ni
            for j=1:ni
                if R(i,j)>=me_R
                    R(i,j)=1;
                else
                    R(i,j)=0;
                end
            end
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%响应峰值的哈希值计算%%与上帧目标框哈希值对比%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        R1_x1=floor(pos(1) - sz(1)/2 + row1-b_weight/2);
        R1_x2=floor(pos(1) - sz(1)/2 + row1+b_weight/2);
        R1_y1=floor(pos(2) - sz(2)/2 + col1-b_height/2);
        R1_y2=floor(pos(2) - sz(2)/2 + col1+b_height/2);
        if R1_x1<=1
            R1_x1=1;
        end
        if R1_y1<=0
            R1_y1=0;
        end
        if R1_x2>=AA(1);
            R1_x2=AA(1)-1;
        end 
        if R1_y2>=AA(2);
            R1_y2=AA(2)-1;
        end  
        R1=im(R1_x1:R1_x2,R1_y1:R1_y2);%   
        R1=imresize(R1,[nr,nr]);
        R1=dct2(R1);
        me_R1=mean(mean(R1));
        ni=8;
        for i=1:ni
            for j=1:ni
                if R1(i,j)>=me_R1
                    R1(i,j)=1;
                else
                    R1(i,j)=0;
                end
            end
        end
        re=uint8(double(R)-double(R1));
        nums1=0;
        for i=1:ni
            for j=1:ni
                if re(i,j)~=0
                    nums1=nums1+1;
                end
            end
        end
        nums1;
        if nums1<5
            row=row1;
            col=col1;
        else
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%候选目标框的哈希值计算%%%%%%%%%%%%%%%%%%%%%%%%%%%候选框的起始位置？？？%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for xii=1:length(m)
                %region=im(floor(pos(1)- row +m(xii)-b_weight/2):floor(pos(1)- row+m(xii)+b_weight/2),floor(pos(2)- col+n(xii)-b_height/2):floor(pos(2)- col+n(xii)+b_height/2));
                r1_x1=floor(pos(1)- sz(1)/2+m(1)-b_weight/2);
                r1_x2=floor(pos(1)- sz(1)/2+m(1)+b_weight/2);
                r1_y1=floor(pos(2)- sz(2)/2+n(1)-b_height/2);
                r1_y2=floor(pos(2)- sz(2)/2+n(1)+b_height/2);
                if r1_x1<=0
                    r1_x1=1;
                end
                if r1_y1<=0
                    r1_y1=1;
                end
                if r1_x2>=AA(1);
                    r1_x2=AA(1)-1;
                end 
                if r1_y2>=AA(2);
                    r1_y2=AA(2)-1;
                end  
                region=im(r1_x1:r1_x2,r1_y1:r1_y2);%图中切出的候选目标框
                region=imresize(region,[nr,nr]);
                region=dct2(region);
                me=mean(mean(region));
                for i=1:ni
                    for j=1:ni
                        if region(i,j)>=me
                            region(i,j)=1;
                        else
                            region(i,j)=0;
                        end
                    end
                end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%目标框与多个峰值响应的哈希值对比%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                re=uint8(double(R)-double(region));
                nums(xii)=0;
                for i=1:ni
                    for j=1:ni
                        if re(i,j)~=0
                            nums(xii)=nums(xii)+1;
                        end
                    end
                end
                N=nums;
            end  
            [x1,y1]=find(N==min(N));%找到10个最大值中哈希距离最小的坐标
            if length(x1)~=1
                for i=1:length(x1)
                    t=y1(i);
                    dd=sqrt((m(t)-row1)^2+(n(t)-col1)^2);
                    if dd ==min(dd)
                        ddd=t
                    end
                end
                row=m(t);
                col=n(t);
                %row=row1;
                %col=col1;
            else
                nums1=min(N);
                row=m(y1);
                col=n(y1);
            end
        end
        rho = rho/(1+exp(nums1/8));			        %the learning parameter \rho in Eq.(12)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        pos(1) = pos(1) - sz(1)/2 + row;
        pos(2) = pos(2) - sz(2)/2 + col;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        contextprior = get_context(im, pos, sz, window);
        conftmp = real(ifft2(Hstcf.*fft2(contextprior))); 
        maxconf(frame-1)=max(conftmp(:));
        %% update scale by Eq.(15)
%        if (mod(frame,num+2)==0)
%            scale_curr = 0;
%            for kk=1:num
%               scale_curr = scale_curr + sqrt(maxconf(frame-kk)/maxconf(frame-kk-1))
%            end            
%            scale = (1-lambda)*scale+lambda*(scale_curr/num);%update scale
%        end  
        %%
    end	
	%% update the spatial context model h^{sc} in Eq.(9)
   	contextprior = get_context(im, pos, sz, window); 
    hscf = conff./(fft2(contextprior)+eps);% Note the hscf is the FFT of hsc in Eq.(9)
    %% update the spatio-temporal context model by Eq.(12)
    if frame == 1,  %first frame, initialize the spatio-temporal context model as the spatial context model
		Hstcf = hscf;
	else
		%update the spatio-temporal context model H^{stc} by Eq.(12)
		Hstcf = (1 - rho) * Hstcf + rho * hscf;% Hstcf is the FFT of Hstc in Eq.(12)
    end
    %% visualization
    target_sz([2,1]) = target_sz([2,1])*scale;% update object size
	rect_position = [pos([2,1]) - (target_sz([2,1])/2), (target_sz([2,1]))];  
    imagesc(uint8(img))
    colormap(gray)
    rectangle('Position',rect_position,'LineWidth',4,'EdgeColor','r');
    A(frame,:)=rect_position;
    hold on;
    text(5, 18, strcat('#',num2str(frame)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
    set(gca,'position',[0 0 1 1]); 
    pause(0.001); 
    hold off;
    drawnow;    
end
A=A(:,1:2);
