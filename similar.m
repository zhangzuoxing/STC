clear all;
close all;
clc;
img=imread('0001_1.bmp');
img2=imread('0001_2.bmp');
imshow(img)
figure,imshow(img2);

tmp=rgb2gray(img);
tmp2=rgb2gray(img2);

img_re=imresize(tmp,[8 8]);
img_re2=imresize(tmp2,[8 8]);

img_re=uint8(double(img_re)/4);
img_re2=uint8(double(img_re2)/4);

me=mean(mean(img_re));
me2=mean(mean(img_re2));

for i=1:8
    for j=1:8
        if img_re(i,j)>=me
            img_re(i,j)=1;
        else
            img_re(i,j)=0;
        end
        
        if img_re2(i,j)>=me2
            img_re2(i,j)=1;
        else
            img_re2(i,j)=0;
        end
                        
    end
end

re=uint8(double(img_re)-double(img_re2));

num=0;
for i=1:8
    for j=1:8
        if re(i,j)~=0
            num=num+1;
        end
    end
end
num