function [y_true,y,y_var,fit_Bay,PosteriorMdl,PriorMdl]=bayesls_rgrn(X_train,y_train,X_test,y_true)
    %% Bayesian Least Squares
    switch size(y_train,2)
        case 1
            B=1.497854;
            A=2.95936659398203e-11;
            run=1;
            c=0;
             %% Alpha and Gamma values
            while run
                run=0;
                t=y_train+(1/B)*randn(size(y_train));
    
                Phi_N=[ones(size(X_train)) X_train];
                Phi_test=[ones(size(X_test)) X_test];
                e=[];
                
    
                % Mean and Covariance matrix of the posterior distribution
                S_N_inv=A*eye(size(Phi_N,2))+B*(Phi_N)'* Phi_N;
                S_N=inv(S_N_inv);
                m_N=B*S_N*Phi_N'*t;
                while c<3 
                    e=eig(B*Phi_N'*Phi_N);
                    gamma=zeros(size(A));
                    for i=1:length(e) %check this  
                        gamma=gamma+e(i)/(e(i)+A);
                    end
                    A_=gamma/(m_N'*m_N);
                %     B=0;
                    N=length(Phi_N);
                    beta_inv=0;
                    for j=1:length(Phi_N)
                        beta_inv= beta_inv+(1/(N-gamma)*(t(j)-Phi_N(j,:)*m_N).^2);
                    end
                    B_=  1/beta_inv;
                    c=c+1
                    A_new= A_
                    B_new= B_
                    if (isnan(A_new)) || (isnan(B_new))
                        break
                    else
                    dm_A=  A_new-A_;
                    dm_B=  B_new-B_;
                    if abs(dm_A)<=10^-4
                        if abs(dm_B)<=10^-4
                            fprintf('Convergence reached! \n');
                            break
                            
                        end
                    end
                    end
                end
                if (~(isnan(A_new))&&(A~=A_new)) || (~(isnan(B_new))&&(B~=B_new))
                    run=run+1
                        A=A_new
                        B=B_new
                    
                end
            
            end
           B=1.497854;
           A=2.95936659398203e-11;
           t=y_train+(1/B)*randn(size(y_train));
                
    
            % Mean and Covariance matrix of the posterior distribution
            S_N_inv=A*double(eye(size(Phi_N,2)))+B*(double(Phi_N)'* double(Phi_N));
            S_N=inv(double(S_N_inv));
            m_N=B*double(S_N)*double(Phi_N)'*double(t);
            % Mean and Variances of the posterior predictive distribution
            y=double(Phi_test)*double(m_N);
            y_var=1/B+sum(double(Phi_test)*double(S_N).*double(Phi_test),2);
            % noise=y_true-(-0.3+X_test*[0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5]);
            

        case 2
            B=[1.497854,1.497854];
            A=[2.95936659398203e-11,2.95936659398203e-11];
            %% Splitting into Train and Test sets
            % X_train=feat(1:(size(feat,1)/2)-1,:);
            % y_train=feat_out(1:(size(feat,1)/2)-1,:);
            t(:,1)=y_train(:,1)+(1/B(:,1))*randn(size(y_train(:,1)));
            % X_test=feat((size(feat,1)/2):size(feat,1),:);
            % y_true=feat_out(size(feat,1)/2:size(feat,1),:);
            
            
            Phi_N=[ones(size(X_train)) X_train];
            Phi_test=[ones(size(X_test)) X_test];
            
            % Mean and Covariance matrix of the posterior distribution
            S_N_inv(:,:,1)=A(:,1)*double(eye(size(Phi_N,2)))+B(:,1)*(double(Phi_N)'*double(Phi_N));
            S_N(:,:,1)=inv(double(S_N_inv(:,:,1)));
            m_N(:,1)=B(:,1)*double(S_N(:,:,1))*double(Phi_N)'*double(t(:,1));
            
            % Mean and Variances of the posterior predictive distribution
            y(:,1)=double(Phi_test)*double(m_N(:,1));
            y_var(:,1)=1/B(:,1)+sum(double(Phi_test)*double(S_N(:,:,1)).*double(Phi_test),2);
            % noise=y_true-(-0.3+X_test*[0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5]);
            
            %% Alpha and Gamma values
            e(:,1)=eig(B(:,1)*double(Phi_N')*double(Phi_N));
            gamma=zeros(size(A));
            for i=1:length(e(:,1)) %check this  
             gamma(:,1)=gamma(:,1)+e(i,1)/(e(i,1)+A(:,1));
            end
            A(:,1)=gamma(:,1)/dot(double(m_N(:,1))',double(m_N(:,1)));
        %     B=0;
            N=length(Phi_N);
            beta_inv(:,1)=1/B(:,1);
            for j=1:length(Phi_N)
                beta_inv(:,1)=beta_inv(:,1)+(1/(N-gamma(:,1))*(t(j,1)-Phi_N(j,:)*m_N(:,1)).^2);
            end
            %% Second Output Case
            %% Splitting into Train and Test sets
            % X_train=feat(1:(size(feat,1)/2)-1,:);
            % y_train=feat_out(1:(size(feat,1)/2)-1,:);
            t(:,2)=y_train(:,2)+(1/B(:,2))*randn(size(y_train(:,2)));
            % X_test=feat((size(feat,1)/2):size(feat,1),:);
            % y_true=feat_out(size(feat,1)/2:size(feat,1),:);
            
            
            Phi_N=[ones(size(X_train)) X_train];
            Phi_test=[ones(size(X_test)) X_test];
            
            % Mean and Covariance matrix of the posterior distribution
            S_N_inv(:,:,2)=A(:,2)*double(eye(size(Phi_N,2)))+B(:,2)*(double(Phi_N)'*double(Phi_N));
            S_N(:,:,2)=inv(double(S_N_inv(:,:,2)));
            m_N(:,2)=B(:,2)*double(S_N(:,:,2))*double(Phi_N)'*double(t(:,2));
            
            % Mean and Variances of the posterior predictive distribution
            y(:,2)=double(Phi_test)*double(m_N(:,2));
            y_var(:,2)=1/B(:,2)+sum(double(Phi_test)*double(S_N(:,:,2)).*double(Phi_test),2);
            % noise=y_true-(-0.3+X_test*[0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5;0.5]);
            
            %% Alpha and Gamma values
            e(:,2)=eig(B(:,2)*double(Phi_N')*double(Phi_N));
%             gamma=0;
            for i=1:length(e(:,2)) %check this  
             gamma(:,2)=gamma(:,2)+e(i,2)/(e(i,2)+A(:,2));
            end
            A(:,2)=gamma(:,2)/dot(double(m_N(:,2))',double(m_N(:,2)));
        %     B=0;
            N=length(Phi_N);
            beta_inv(:,2)=1/B(:,2);
            for j=1:length(Phi_N)
                beta_inv(:,2)=beta_inv(:,2)+(1/(N-gamma(:,2))*(t(j,2)-Phi_N(j,:)*m_N(:,2)).^2);
            end
    end

%     figure(6)
%     err=0.025*sqrt(y_var/N);
%     errorbar(y,err),xlabel('time'),ylabel('force')
%     title('Posterior Outcome with 95% Confidence Interval');
    x=1:size(y_true,1);                        % Mean Of All Experiments At Each Value Of tx=
    ySEM = sqrt(y_var/N);                             % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM', CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of \x'
    figure
    plot(x,y')                                      % Plot Mean Of All Experiments
    hold on
    switch size(y,2)
        case 1
            plot(x,yCI95+y')                                % Plot 95% Confidence Intervals Of All Experiments
        case 2
            plot(x,yCI95+y(:,1)')
            hold on
            plot(x,yCI95+y(:,2)')
    end
    hold off
    grid
    title('Posterior Outcome with 95% Confidence Interval');
                
    figure
    plot(y_true,'r'),xlabel('time'),ylabel('force'), hold on, plot(y,'b'),xlabel('time'),ylabel('force'),legend('Output','ModelOutput')
    title('Bayesian Least Squares Regression');
    
    y_pred=y;
    % err=immse(Predicted_Out,feat_out)
    RMSE_Bay=sqrt(mean((y_pred-y_true).^2));
    Mean=mean(y_true);
    
    
    SSR_Bay=sum((y_pred-Mean).^2,1);
    SST_Bay=sum((y_true-Mean).^2,1);
    
    R_squared_Bay= SSR_Bay./SST_Bay;
    fit_Bay=R_squared_Bay.*100;
    
    fprintf('Goodness of fit for Bayesian Least Squares Regression: %f%%, %f%% \n',fit_Bay)
    fprintf('RMSE for Bayesian Least Squares Regression: %f, %f \n',RMSE_Bay)
    A
    B
    
     %% Bayesian Least Squares Using Matlab in-built functions
    
    switch size(X_train,2)
        case 12
            p = 12;
            VarNames = ["s-EMG1" "s-EMG2" "s-EMG3" "s-EMG4" "s-EMG5" "s-EMG6" "s-EMG7" "s-EMG8" "s-EMG9" "s-EMG10" "s-EMG11" "s-EMG12"];%12 coefficient names
        case 10
            p = 10;
            VarNames = ["s-EMG1" "s-EMG2" "s-EMG3" "s-EMG4" "s-EMG5" "s-EMG6" "s-EMG7" "s-EMG8" "s-EMG11" "s-EMG12"];%10 coefficient names
        case 2
            p = 2;
            VarNames = ["s-EMG11" "s-EMG12"];%12 coefficient names
    end

    PriorMdl = bayeslm(p,'ModelType','conjugate','V',double(10^11*eye(p)),'VarNames',VarNames);%1/alpha=10^11
    summarize(PriorMdl)
        % PriorMdl.A = 12;
        % PriorMdl.B = 1;%1/beta=1 Scaling

    switch size(y_train,2)
        case 1
            PosteriorMdl = estimate(PriorMdl,X_train,y_train);
            summarize(PosteriorMdl)

             % title('Marginal posterior distributions of the intercept, regression coefficients, and disturbance variance')
            plot(PriorMdl,PosteriorMdl);%Marginal posterior distributions of the intercept, regression coefficients, and disturbance variance.
    
            yF = forecast(PosteriorMdl,X_test);
            figure
            plot(yF,'r'),xlabel('time'),ylabel('force'), hold on, plot(y_true,'y'),xlabel('time'),ylabel('force'),legend('ModelOutput','Output')
            title('Bayesian Linear System Identification using built-in Matlab Function')
        case 2
            PosteriorMdl(1,:) = estimate(PriorMdl,X_train,y_train(:,1));
            summarize(PosteriorMdl(1,:))

            plot(PriorMdl,PosteriorMdl(1,:));

            yF(1,:) = forecast(PosteriorMdl(1,:),X_test);
            figure
            plot(yF(1,:),'r'),xlabel('time'),ylabel('force'), hold on, plot(y_true(:,1),'y'),xlabel('time'),ylabel('force'),legend('ModelOutput','Output')
            title({'Bayesian Linear System Identification using built-in Matlab Function', 'corresponding to middle finger force output'})

            PosteriorMdl(2,:) = estimate(PriorMdl,X_train,y_train(:,2));
            summarize(PosteriorMdl(2,:))
            
            plot(PriorMdl,PosteriorMdl(2,:));

            yF(2,:) = forecast(PosteriorMdl(2,:),X_test);
            figure
            plot(yF(2,:),'r'),xlabel('time'),ylabel('force'), hold on, plot(y_true(:,2),'y'),xlabel('time'),ylabel('force'),legend('ModelOutput','Output')
            title({'Bayesian Linear System Identification using built-in Matlab Function', 'corresponding to index finger force output'})
    end
    
   
    

end 
%  mu = 0; sigma = sqrt(1/A);
%  x_ = mu-(4*sigma):0.01:mu+(4*sigma);
%  y_pdf = normpdf(x_,mu,sigma);
%  figure(1);plot(x_,y_pdf,'--r')
%  
%  mu = m_N; sigma = sqrt(S_N);
% x = mu-(4*sigma):0.01:mu+(4*sigma);
%  y_posterior_pdf = normpdf(x,mu,sigma);
%  figure(3);plot(x,y_posterior_pdf)
% B=750.0;%beta
% A=0.00000005;%alpha
% B= 1.498036;
% % A=2.95932414567089e-11;
% % B=750.0;%beta
% % A=0.00000005;%alpha
% % A=2.33012142292914e-11;%beta
% A=2.95932414567089e-11;
% B=1.416162;%alpha