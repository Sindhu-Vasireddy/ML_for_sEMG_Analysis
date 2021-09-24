
function [M_Out,RMSE_mvrgrn,fit_mvrgrn]=multivariate_rgrn(in_train,out_train,in_test,out_test)
%% Multi-Variate Regression using Covariance-Weighted Least Squares 
    [beta,Sigma] = mvregress(in_train,out_train,'algorithm','cwls');
    M_Out=in_test*beta;


    RMSE_mvrgrn=sqrt(mean((M_Out-out_test).^2));


    Mean_mvrgrn=mean(out_test);

    SSR_mvrgrn=sum((M_Out-Mean_mvrgrn).^2,1);
    SST_mvrgrn=sum((out_test-Mean_mvrgrn).^2,1);

    R_squared_mvrgrn= SSR_mvrgrn./SST_mvrgrn;
    fit_mvrgrn=R_squared_mvrgrn.*100;
    figure
    plot(M_Out,'y'),xlabel('time'),ylabel('force'), hold on, plot(out_test,'b'),xlabel('time'),ylabel('force'),legend('ModelOutput','Output')
    title('Multi-Variate Regression using Covariance-Weighted Least Squares')
            
    fprintf('RMSE for Multi Variate Regression: %f, %f \n',RMSE_mvrgrn);
    fprintf('Goodness of fit for Multi Variate Regression: %f%%, %f%% \n',fit_mvrgrn);
% 
%     return M_Out,RMSE_mvrgrn,fit_mvrgrn;

end