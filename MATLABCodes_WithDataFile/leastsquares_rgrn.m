
function [Predicted_Out,RMSE,fit]= leastsquares_rgrn(in_train,out_train,in_test,out_test)
%% Least Squares
    Y=out_train;
    PHI=in_train;
    Beta=((PHI'*PHI)\PHI')*Y;
    
    PHI=in_test; 
    Predicted_Out=PHI*Beta;


    % err=immse(Predicted_Out,feat_out)
    RMSE=sqrt(mean((Predicted_Out-out_test).^2));
    Mean=mean(out_test);
    y_pred=Predicted_Out;
    y_true=out_test;
    
    SSR =sum((y_pred-Mean).^2,1);
    SST =sum((y_true-Mean).^2,1);
    
    R_squared = SSR./SST ;
    fit =R_squared.*100;
    

    figure
    plot(Predicted_Out,'r'),xlabel('time'),ylabel('force'),hold on, plot(out_test,'y'),xlabel('time'),ylabel('force'),legend('ModelOutput','Output')
    title('Least Squares Linear Regression')
        
    fprintf('RMSE for Least Squares Regression: %f, %f\n',RMSE);
    fprintf('Goodness of fit for Least Squares Regression: %f%%, %f%% \n',fit);
%     return Predicted_Out,RMSE,fit;
    
end