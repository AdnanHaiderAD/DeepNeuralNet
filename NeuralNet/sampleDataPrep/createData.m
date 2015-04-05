function[trainingInput, trainingLabel,testInput,testLabel]= createData(x,y,f)
[r,c] = size(x);
count = 1;
for i = 1 : 4: r
    for j = 1 : 5 : r
        trainingInput(count,:) = [x(i) y(j)];
        trainingLabel (count)  = f(i,j);
        count= count+1;
    end
end
count = 1;
for i= 2 : 4:r
    for j = 2:5:r
        testInput(count,:) = [x(i) y(j)];
        testLabel(count) = f(i, j);
        count=count+1;
    end
end
