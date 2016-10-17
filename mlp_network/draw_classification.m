function [Color,PredictedColor] = draw()
    fileID = fopen('train.txt','r');
    formatSpec = '%f';
    C =(fscanf(fileID,formatSpec));
    Input = C(1:size(C)/2)';
    [X,Y] = findXY(Input);
    Color = C(size(C)/2 + 1 : 3*size(C)/4)';
    PredictedColor = C(3*size(C)/4 + 1 : size(C))';
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
    
    drawScatter(MarkerColors, size(C,1)/4, X, Y, Color, 'o');
    drawScatter(MarkerColors, size(C,1)/4, X, Y, PredictedColor, '.');
 
    fileID = fopen('test.txt','r');
    C1 =(fscanf(fileID,formatSpec));
    Input1 = C1(1:2*size(C1)/3)';
    [X1,Y1] = findXY(Input1);
    Color1 = C1(2*size(C1)/3 + 1 : size(C1))';
    fclose(fileID);
    
    drawScatter(MarkerColors, size(C1, 1)/3, X1, Y1, Color1, '.');
    hold off;
end

function [X,Y] = findXY(Input)
    n = size(Input,2);
    X = zeros(1,n/2);
    Y = zeros(1,n/2);
    i = 1;
    index = 1;
    while (i <= n)
        X(index) = Input(1,i);
        Y(index) = Input(1,i+1);
        i = i + 2;
        index = index + 1;
    end
end

function [] = drawScatter(MarkerColors, sizeT, X, Y, Color, style)
    for index = 1 : size(MarkerColors,1)
        Data = zeros(2, sizeT);
        last = 0;
        for i = 1 : sizeT
            j = Color(1,i);
            if j == index
                last = last + 1;
                Data(1,last) = X(1,i);
                Data(2,last) = Y(1,i);
            end
        end
        PlotData = Data(:,1:last);
        scatter(PlotData(1,:),PlotData(2,:), style, 'MarkerEdgeColor', MarkerColors(index,1:3));
        hold on;
    end

end
