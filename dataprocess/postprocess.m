clear all
pred_path='';
post_path2='';
mkdir(post_path2)
pred_res=dir(fullfile(pred_path,'*.png'));

name1=pred_res(1).name;
a1=find(name1=='_');
id1=name1(1:a1(3)-1);
id_p{1,1}=name1;
id_l=1;
id_d=1;
 for i=2:length(pred_res)
% i=3
    name=pred_res(i).name;
   a2=find(name=='_');
   id_2=name(1:a2(3)-1);
    if length( id_2)==length(id1)
        if id_2==id1
            id_p{id_l,id_d+1}=name;
            id_d=id_d+1;
        else
             id_d=1;
             id_l=id_l+1;
             id_p{id_l,id_d}=name;
             name1=name;
            a1=find(name1=='_');
            id1=name1(1:a1(3)-1);
        end
        else
             id_d=1;
             id_l=id_l+1;
             id_p{id_l,id_d}=name;
             name1=name;
            a1=find(name1=='_');
            id1=name1(1:a1(3)-1);
    end
      
end
b=size(id_p);
for j=1:b(1)
    
    n=id_p(j,:);
    n(cellfun(@isempty,n))=[];
    
    if length(n)==1
        pred_image=imread(fullfile(pred_res_path,n{1}));
        name_pred=n{1};
        a=find(name_pred=='_');
        name_pred=[name_pred(1:a(3)-1),'_gt.png'];
        imwrite(pred_image,fullfile(post_path2,name_pred))
    elseif length(n)==2
%     else
        pred_image1=imread(fullfile(pred_res_path,n{1}));
        pred_image2=imread(fullfile(pred_res_path,n{2}));
        kk1=find(pred_image1==255);
        k2=find(pred_image2==255);
       m=setdiff(kk1,k2);
       pred_image(m)=0;
        name_pred=n{1};
        a=find(name_pred=='_');
        name_pred=[name_pred(1:a(3)-1),'_gt.png'];
       imwrite(pred_image1,fullfile(post_path2,name_pred))
    else
         pred_image1=imread(fullfile(pred_res_path,n{1}));
         pred_image2=imread(fullfile(pred_res_path,n{2}));
         pred_image3=imread(fullfile(pred_res_path,n{3}));
        k1=find(pred_image1==255);
        k2=find(pred_image2==255);        
        k3=find(pred_image3==255);     
        b=intersect(k1,k2);
        b=intersect(b,k3);  
        kk1=setdiff(k1,b);
        kk2=setdiff(k2,b);
        kk3=setdiff(k3,b);
        b12=intersect(kk1,k2);
        b23=intersect(k2,k3);
        b13=intersect(kk1,k3);
        pred_image=uint8(zeros(400,400,1));
        pred_image(b)=255;
        pred_image(b12)=255;
        pred_image(b23)=255;
        pred_image(b13)=255;
        name_pred=n{1};
        a=find(name_pred=='_');
        name_pred=[name_pred(1:a(3)-1),'_gt.png'];      
        imwrite(pred_image,fullfile(post_path2,name_pred))
    end
     
end



