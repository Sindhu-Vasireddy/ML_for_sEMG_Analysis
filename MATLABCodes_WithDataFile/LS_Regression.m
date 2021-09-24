% v=size(feat,2);
%% Train(1,3,4,6) and Test(2,5) Set Split
out_train=double(feat_out(1:1350,:));
in_train=double(feat(1:1350,:));%70% train
out_test=double(feat_out(1351:end,:));
in_test=double(feat(1351:end,:));%30% test


%%
v=10;
q=2;
% switch v
%     case 2
%         switch q 
%             case 1
%                 [M_Out,RMSE_mvrgrn,fit_mvrgrn]= multivariate_rgrn(in_train(:,end-1:end),out_train(:,4),in_test(:,end-1:end),out_test(:,4));
%                 [Predicted_Out,RMSE,fit]= leastsquares_rgrn(in_train(:,end-1:end),out_train(:,4),in_test(:,end-1:end),out_test(:,4));
%                 [y_true,y,y_var,fit_Bay,PosteriorMdl,PriorMdl]=bayesls_rgrn(in_train(:,end-1:end),out_train(:,4),in_test(:,end-1:end),out_test(:,4));
%                 [fit_lstm,RMSE_lstm]=lstm_rgrn(in_train(:,end-1:end),out_train(:,4),in_test(:,end-1:end),out_test(:,4))

%             case 2
%                 [M_Out,RMSE_mvrgrn,fit_mvrgrn]= multivariate_rgrn(in_train(:,end-1:end),out_train(:,3:4),in_test(:,end-1:end),out_test(:,3:4));
%                 [Predicted_Out,RMSE,fit]= leastsquares_rgrn(in_train(:,end-1:end),out_train(:,3:4),in_test(:,end-1:end),out_test(:,3:4));
%                 [y_true,y,y_var,fit_Bay,PosteriorMdl,PriorMdl]=bayesls_rgrn(in_train(:,end-1:end),out_train(:,3:4),in_test(:,end-1:end),out_test(:,3:4));
%                 [fit_lstm,RMSE_lstm]=lstm_rgrn(in_train(:,end-1:end),out_train(:,3:4),in_test(:,end-1:end),out_test(:,3:4))
% 
%         end
%     case 10
%         switch q 
%             case 1
%                 [M_Out,RMSE_mvrgrn,fit_mvrgrn]=multivariate_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,4));
%                 [Predicted_Out,RMSE,fit]= leastsquares_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,4));
%                 [y_true,y,y_var,fit_Bay,PosteriorMdl,PriorMdl]=bayesls_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,4));
%                 [fit_lstm,RMSE_lstm]=lstm_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,4));

%             case 2
%                 [M_Out,RMSE_mvrgrn,fit_mvrgrn]=multivariate_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,3:4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,3:4));
%                 [Predicted_Out,RMSE,fit]= leastsquares_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,3:4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,3:4));
%                 [y_true,y,y_var,fit_Bay,PosteriorMdl,PriorMdl]=bayesls_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,3:4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,3:4));
                [fit_lstm,RMSE_lstm]=lstm_rgrn([in_train(:,1:8) in_train(:,11:12)],out_train(:,3:4),[in_test(:,1:8) in_test(:,11:12)],out_test(:,3:4))
%         end
%     case 12
%         switch q 
%             case 1
%                 [M_Out,RMSE_mvrgrn,fit_mvrgrn]=multivariate_rgrn(in_train,out_train(:,4),in_test,out_test(:,4));
%                 [Predicted_Out,RMSE,fit]= leastsquares_rgrn(in_train,out_train(:,4),in_test,out_test(:,4));
%                 [y_true,y,y_var,fit_Bay,PosteriorMdl,PriorMdl]=bayesls_rgrn(in_train,out_train(:,4),in_test,out_test(:,4));
%                 [fit_lstm,RMSE_lstm]=lstm_rgrn(in_train,out_train(:,4),in_test,out_test(:,4))
% 
%             case 2
%                 [M_Out,RMSE_mvrgrn,fit_mvrgrn]=multivariate_rgrn(in_train,out_train(:,3:4),in_test,out_test(:,3:4));
%                 [Predicted_Out,RMSE,fit]= leastsquares_rgrn(in_train,out_train(:,3:4),in_test,out_test(:,3:4));
%                 [y_true,y,y_var,fit_Bay,PosteriorMdl,PriorMdl]=bayesls_rgrn(in_train,out_train(:,3:4),in_test,out_test(:,3:4));
%                 [fit_lstm,RMSE_lstm]=lstm_rgrn(in_train,out_train(:,3:4),in_test,out_test(:,3:4))                
% 
%         end
% end

        