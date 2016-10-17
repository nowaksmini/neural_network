fileID = fopen('train.txt','r');
formatSpec = '%f';
C =(fscanf(fileID,formatSpec));
X = C(1:size(C)/4)';
Y = C(size(C)/4 + 1 : size(C)/2)';
Color = C(size(C)/2 + 1 : 3*size(C)/4)';
figure % new figure
fclose(fileID);

MarkerColors = [rand(), rand(), rand()];

for i = 1 : size(C)/4
      j = Color(1,i);
      if j > size(MarkerColors,1)
          TempColors = zeros(j,3);
          for index = 1 : size(MarkerColors,1)
              TempColors(index, 1 : 3) = MarkerColors(index, 1 : 3);
          end
          for index = size(MarkerColors,1) : j
              TempColors(index, 1 : 3) = [rand(), rand(), rand()];
          end
          MarkerColors = TempColors;
      end
end

for index = 1 : size(MarkerColors)
    Data = zeros(2, size(C, 1)/4);
    last = 0;
    for i = 1 : size(C)/4
        j = Color(1,i);
        if j == index
            last = last + 1;
            Data(1,last) = X(1,i);
            Data(2,last) = Y(1,i);
        end
    end
    PlotData = Data(:,1:last);
    scatter(Data(1,:),Data(2,:),'MarkerEdgeColor', MarkerColors(index,1:3));
    hold on;
end

hold off;

