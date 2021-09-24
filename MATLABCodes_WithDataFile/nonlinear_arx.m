function [sys_2,fit_nlarx,RMSE_nlarx]=nonlinear_arx(Di,ze,zv) 
    %% Non-Linear ARX
    % opt=nlarxOptions('Display','on');
    switch size(Di.InputData,2)
        case 12
            switch size(Di.y,2)
                case 1
                    sys_2=nlarx(Di, "na", 2, "nb", [6 1 1 0 0 0 1 0 0 0 0 0], "nk", [0 0 0 1 1 1 0 1 1 1 1 1])
                case 2
                    sys_2=nlarx(Di, "na", [2 2;2 2], "nb", [6 1 1 0 0 0 1 0 0 0 0 0;6 1 1 0 0 0 1 0 0 0 0 0], "nk", [0 0 0 1 1 1 0 1 1 1 1 1;0 0 0 1 1 1 0 1 1 1 1 1])
            end
        case 10
            switch size(Di.y,2)
                case 1
                    sys_2=nlarx(Di, "na", 2, "nb", [6 1 1 0 0 0 1 0 0 0], "nk", [0 0 0 1 1 1 0 1 1 1])
                case 2
                    sys_2=nlarx(Di, "na", [2 2;2 2], "nb", [6 1 1 0 0 0 1 0 0 0;6 1 1 0 0 0 1 0 0 0], "nk", [0 0 0 1 1 1 0 1 1 1;0 0 0 1 1 1 0 1 1 1])
            end            
        case 2
            switch size(Di.y,2)
                case 1
                    sys_2=nlarx(Di, "na", 2, "nb", [0 0], "nk", [0 0])
                case 2
                    sys_2=nlarx(Di, "na", [1 2;2 1], "nb", [3 10;3 10], "nk", [0 0;0 0])
            end
    end
    
    fig=figure
    compare(Di,sys_2)
    Number_of_inputs=size(Di.InputData,2);
    Number_of_outputs=size(Di.y,2);
    title(['Non-Linear ARX System Identification with:' num2str(Number_of_inputs) 'input(/s)'  num2str(Number_of_outputs) 'output(/s)'] )
    % fig = openfig('arx_nlarx_compare.fig');
    D=get(gca,'Children');
    switch size(Di.y,2)
        case 1
            YData_nlarx(:,1)=get(D(1).Children,'YData');
        case 2
            YData_nlarx(:,1)=get(D(1).Children,'YData');
            YData_nlarx(:,2)=get(D(2).Children,'YData');
    end

    
    
    y_pred=YData_nlarx;
%     y_true=feat_out;
    y_true=Di.y;
    % err=immse(Predicted_Out,feat_out)
    RMSE_nlarx=sqrt(mean((y_pred-y_true).^2));
    Mean=mean(y_true);
    
    SSR_nlarx=sum((y_pred-Mean).^2,1);
    SST_nlarx=sum((y_true-Mean).^2,1);
    
    R_squared_nlarx= SSR_nlarx./SST_nlarx;
    fit_nlarx=R_squared_nlarx.*100;
    
    fprintf('Goodness of fit for Non-Linear ARX Regression: %f%%, %f%% \n',fit_nlarx)
    fprintf('Root Mean Square Error for Linear ARX Regression: %f, %f \n',RMSE_nlarx)
end