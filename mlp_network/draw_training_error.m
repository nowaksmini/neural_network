fileID = fopen('training-erros.txt','r');
formatSpec = '%f';
A =(fscanf(fileID,formatSpec))';
B = 1: size(A,2);
figure % new figure
ax1 = subplot(1,1,1); % top subplot
plot(ax1,B,A);
title(ax1,'Learning error')
xlabel(ax1,'Iteration')
ylabel(ax1,'Error')
fclose(fileID);