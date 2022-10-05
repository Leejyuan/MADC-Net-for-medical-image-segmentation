clear all
warning off
image_path='';
label_path='';
image_path2='';
label_path2='';
mkdir(image_path2)
mkdir(label_path2)
files=dir(fullfile(image_path,'*.png'));
image_name=files(1).name;
a=find(image_name=='_');
id1=image_name(1:a(1)-1);
j=1;
n=1;
id_n{j,n}=image_name;

for i=2:length(files)
    image_name=files(i).name;
    a=find(image_name=='_');
    id=image_name(1:a(1)-1);
    if length(id)==length(id1)
        if id==id1
            n=n+1;
            id_n{j,n}=image_name;
        else
            j=j+1;
            n=1;
            id_n{j,n}=image_name;
            id1=id;
        end
    else
            j=j+1;
            n=1;
            id_n{j,n}=image_name;
            id1=id;
    end

end

for j=1:size(id_n,1)
   n=id_n(j,:);
    n(cellfun(@isempty,n))=[];
    for jj=2:length(n)
     image_m=imread(fullfile(image_path,n{jj}));
     image_b=imread(fullfile(image_path,n{jj-1}));
     image(:,:,1)=image_b;
     image(:,:,2)=image_m;
     image(:,:,3)=image_m;
     imwrite(image,[image_path2,'\',n{jj}])
     label_m=imread(fullfile(label_path,n{jj}));
     label_b=imread(fullfile(label_path,n{jj-1}));
     a1=find(label_m==255);
     a2=find(label_b==255);
     b=intersect(a1,a2);
      k1=setdiff(a1,b);
      k2=setdiff(a2,b);
      label_25=uint8(zeros(400,400,1));
      label_25(b)=255;
      label_25(k1)=170;
      label_25(k2)=85;
      imwrite(label_25,[label_path2,'\',n{jj}])
    end
    
end



