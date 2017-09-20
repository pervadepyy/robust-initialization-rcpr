function face = getface( Is,bboxes )
%GETFACELBP Summary of this function goes here
%   Detailed explanation goes here
N=size(Is,1);
face = cell(N,1);
for i=1:N
    img = Is{i};
    [h,w] = size(img);
    x1 = max(1,bboxes(i,1));
    x2 = min(w,bboxes(i,1)+bboxes(i,3)-1);
    y1 = max(1,bboxes(i,2));
    y2 = min(h,bboxes(i,2)+bboxes(i,4)-1);
    
    faceimg = img(y1:y2,x1:x2);
    face{i,1}=faceimg;
end

end

