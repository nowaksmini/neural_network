fileID = fopen('regression_test.txt','r');
formatSpec = '%f';
C =(fscanf(fileID,formatSpec));
X = C(1:size(C)/3)';
Y = C(size(C)/3 + 1:2*size(C)/3)';
CalculatedTraining = C(2 * size(C)/3 + 1:size(C))';
figure % new figure
fclose(fileID);

fileID = fopen('regression.txt','r');
formatSpec = '%f';
C1 =(fscanf(fileID,formatSpec));
X1 = C1(1:size(C1)/2)';
Y1 = C1(size(C1)/2 + 1:size(C1))';
fclose(fileID);
plot(X,CalculatedTraining, 'r*', X1, Y1, 'k.', X,Y, 'b.');
