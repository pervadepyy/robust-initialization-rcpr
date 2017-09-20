function  IsHist  = getlbpHist( Is )
%GETLBP Summary of this function goes here
%   Detailed explanation goes here
N=size(Is,1);
IsHist = cell(N,1);
block = 8;
mapping=getmapping(8,'u2'); 
for k=1:N
    faceimg = Is{k};
    [h, w] = size(faceimg);    
    blockx = floor(w/block);
    blocky = floor(h/block); 
    H = [];
    for i=1:block
        for j=1:block
            blockface = faceimg(blocky*(i-1)+1:blocky*i,blockx*(j-1)+1:blockx*j);
            H1=lbp(blockface,1,8,mapping,'h');
            H = [H ;H1];
        end
    end
    IsHist{k,1}=H;
end
end

