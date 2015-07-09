% WriteMatrix(filename,matrix)

function WriteMatrix(filename,matrix);

f=fopen(filename,'w');
for i=1:size(matrix,1);
    for j=1:size(matrix,2);
        fprintf(f,'%.6f',matrix(i,j));
        if j~=size(matrix,2);
            fprintf(f,',');
        end
    end
    fprintf(f,'\n');
end
fclose(f);
  
end
