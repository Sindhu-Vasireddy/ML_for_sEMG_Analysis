function [sys_1,fit_arx,RMSE_arx]=linear_arx(Di,ze,zv) 
    %% Linear ARX
%     Di=iddata(double(feat_out),double(feat),1/100);
%     ze=iddata(double(feat_out(1:1350,:)),double(feat(1:1350,:)),1/100);%70% train
%     zv=iddata(double(feat_out(1351:end,:)),double(feat(1351:end,:)),1/100);%30% test
    % sys_1=arx(Di, "na", 2, "nb", [6 1 1 1 0 0 1 0 0 0 0 0], "nk", [0 0 0 0 1 1 0 1 1 1 1 1])
    % sys_1=arx(Di, "na", 2, "nb", [6 6 1 3 0 0 1 0 0 0 0 0], "nk", [0 0 0 0 1 1 0 1 1 1 1 1])%74.53 best
    
    % opt=arxOptions('Display','on');
    switch size(Di.InputData,2)
        case 12

            switch size(Di.y,2)
                case 1
                    sys_1=arx(Di, "na", 2, "nb", [6 6 1 3 0 0 1 0 0 0 0 0], "nk", [0 0 0 0 1 1 0 1 1 1 1 1])
                case 2
                    sys_1=arx(Di, "na", [2 2;2 2], "nb", [6 6 1 3 0 0 1 0 0 0 0 0;6 6 1 3 0 0 1 0 0 0 0 0], "nk", [0 0 0 0 1 1 0 1 1 1 1 1;0 0 0 0 1 1 0 1 1 1 1 1])
            end
        case 10
            switch size(Di.y,2)
                case 1
                    sys_1=arx(Di, "na", 2, "nb", [6 6 1 3 0 0 1 0 0 0], "nk", [0 0 0 0 1 1 0 1 1 1])
                case 2
                    sys_1=arx(Di, "na", [2 2;2 2], "nb", [6 6 1 3 0 0 1 0 0 0;6 6 1 3 0 0 1 0 0 0], "nk", [0 0 0 0 1 1 0 1 1 1;0 0 0 0 1 1 0 1 1 1])
            end

        case 2
            switch size(Di.y,2)
                case 1
%                     NN = struc(1:10,1:10,1:10,0:2,0:2);
%                     V = ivstruc(ze,zv,NN);
%                     order = selstruc(V,0);
%                     sys_1=arx(Di, order)
                    sys_1=arx(Di, "na", 1, "nb", [3 10], "nk", [0 0])
                case 2

                    sys_1=arx(Di, "na", [1 0;0 8], "nb", [3 10;1 6], "nk", [0 0;0 1])
            end

    end

    fig=figure
    compare(Di,sys_1)
    Number_of_inputs=size(Di.InputData,2);
    Number_of_outputs=size(Di.y,2);
    title(['Linear ARX System Identification with:' num2str(Number_of_inputs) 'input(/s)'  num2str(Number_of_outputs) 'output(/s)'] )
    
    D=get(gca,'Children');
    switch size(Di.y,2)
        case 1
            YData_arx(:,1)=get(D(1).Children,'YData');
        case 2
            YData_arx(:,1)=get(D(1).Children,'YData');
            YData_arx(:,2)=get(D(2).Children,'YData');
    end
    y_pred=YData_arx;
%     y_true=feat_out;
    y_true=Di.y;
    % err=immse(Predicted_Out,feat_out)
    RMSE_arx=sqrt(mean((y_pred-y_true).^2));
    Mean=mean(y_true);
    
    SSR_arx=sum((y_pred-Mean).^2,1);
    SST_arx=sum((y_true-Mean).^2,1);
    
    R_squared_arx= SSR_arx./SST_arx;
    fit_arx=R_squared_arx.*100;
    
    fprintf('Goodness of fit for Linear ARX Regression: %f%%, %f%% \n',fit_arx)
    fprintf('Root Mean Square Error for Linear ARX Regression: %f, %f \n',RMSE_arx)
end




