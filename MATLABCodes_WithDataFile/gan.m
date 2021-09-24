%% Define Generator
executionEnvironment = "auto";
tr_out=feat_out(:,4);
%% LSTM Regression for Generator
numResponses = 1;
numHiddenUnits = 200;%200
numFeatures = 12;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(150)%150%increasing to 200 showed improvement.
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

lgraphGenerator = layerGraph(layers);
lgraphGenerator = removeLayers(lgraphGenerator,["regressionoutput"]);
dlnetGenerator = dlnetwork(lgraphGenerator);
%% Define Discriminator
%% LSTM Classifier for Discriminator
numFeatures = 1;
numHiddenUnits = 100;
numClasses = 1;
layersDiscriminator = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

lgraphDiscriminator = layerGraph(layersDiscriminator);
lgraphDiscriminator = removeLayers(lgraphDiscriminator,["softmax" "classoutput"]);


dlnetDiscriminator = dlnetwork(lgraphDiscriminator);
%% Training Options
numEpochs = 1000;
miniBatchSize = 324;
learnRate = 0.0002;%0.0002
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
flipFactor = 0.3;%0.5
validationFrequency = 100;

%% loops=length(c)/miniBatchSize;%number of batches per epoch
numValidation = miniBatchSize;%num of batches(B)
numLatentInputs=12%num of channels(C)
timesamples=1

ZValidation = randn(numValidation*numLatentInputs*timesamples,1,'single');%num of channels*num of batches*timesamples
% Matrix to Cell
zval={};
% o={};
i=1;
r=1;
for j=1:length(ZValidation)
    if mod(j,12*1)==0
        l=ZValidation(i:j);
         l_re=reshape(l,[12,1]);
        zval(r,:)={l_re};
        i=i+12*1;
        r=r+1;
    end
end
zval_arr=cell2mat(zval);%27*12*7 cell array

zval_arr_re=reshape(zval_arr,[miniBatchSize,12,1]);
dlZValidation = dlarray(zval_arr_re,'BCT');%27*1*7
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end
%% Parameters for Adam
trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];


%% Train GAN
start = tic;

tot_iter=0;
% Loop over epochs-dlX.
 y_true_f={};
 % o={};
i=1;
r=1;
for j=1:length(tr_out)
    if mod(j,1)==0
       l=tr_out(i:j);
       y_true_f(r,:)={l'};
       i=i+1;
       r=r+1;
    end
end
loops=length(y_true_f)/miniBatchSize;%number of batches per epoch
for epoch = 1:numEpochs
    k=1
    % Loop over mini-batches.
    for iteration=1:loops
%         iteration = iteration + 1;
        % Read mini-batch of data.

        xval_arr=cell2mat(y_true_f(k:k+miniBatchSize-1));%54*1*7 cell array

        xval_arr_re=reshape(xval_arr,[miniBatchSize,1,1]);
        dlX = dlarray(xval_arr_re,'BCT');%54*1*7
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end

        k=k+miniBatchSize
        tot_iter=tot_iter+1;
        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels 'CB' (channel, batch).
        % If training on a GPU, then convert latent inputs to gpuArray.
        Z_train = randn(numValidation*numLatentInputs*timesamples,1,'single');%num of channels*num of batches*timesamples
        % Matrix to Cell
        zval_train={};
        % o={};
        i=1;
        r=1;
        for j=1:length(Z_train)
            if mod(j,1*12)==0
                l=Z_train(i:j);
                l_re=reshape(l,[12,1]);
                zval_train(r,:)={l_re};
                i=i+12*1;
                r=r+1;
            end
        end
        zval_arr_train=cell2mat(zval_train);%27*12,7 array
        
        zval_arr_re_train=reshape(zval_arr_train,[miniBatchSize,12,1]);
        dlZ = dlarray(zval_arr_re_train,'BCT');%27*12*7
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ = gpuArray(dlZ);
        end
        
        
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, tot_iter, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, tot_iter, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input.
        if mod(tot_iter,validationFrequency) == 0 || tot_iter == 1
            % Test Generator.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
%             plot(dlXGeneratedValidation) 
           
        end
        scoreAxes = subplot(1,1,1);
        lineScoreGenerator = animatedline(scoreAxes,'Marker','.','Color','r');
        lineScoreDiscriminator = animatedline(scoreAxes,'Marker','.', 'Color', 'b');
        legend('Generator','Discriminator');
        ylim([0 1])
        xlabel("Iteration")
        ylabel("Score")
        grid on
        
        % Update the scores plot.
        subplot(1,1,1)
        addpoints(lineScoreGenerator,tot_iter,...
            double(gather(extractdata(scoreGenerator))));
        
        addpoints(lineScoreDiscriminator,tot_iter,...
            double(gather(extractdata(scoreDiscriminator))));
         
        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
    end
end
% figure,plot(reshape(extractdata(dltestgen),[324,1]),'r'),hold on,plot(reshape(extractdata(dlX),[324,1]),'g')
% figure,plot(f_unroll,'r'),hold on,plot(d_unroll,'g');
%% Unrolling before plotting
%
% d_re=reshape(extractdata(dlX),[54,7]);
% t_re=reshape(extractdata(dltestgen),[54,7]);
% i=1;
% for j=1:length(d_re)
% d_unroll(i:i+7-1,1)=d_re(j,:)';
% t_unroll(i:i+7-1,1)=t_re(j,:)'
% i=i+7;
% end
% 
% fig = openfig('dum.fig');
% fig = gcf;
% axObjs = fig.Children
% dataObjs = axObjs.Children
% dataObjs
% dataObjs(1,1)
% dataObjs(2,1)
% y_f= dataObjs(2).YData
% y_t= dataObjs(1).YData
% figure(6),plot(y_t)
% figure(7),plot(y_f)

% t_re=reshape(y_t',[54,1,7]);
% t_re=reshape(t_re,[54,7]);
% i=1;
% for j=1:length(t_re)
% t_unroll(i:i+7-1,1)=t_re(j,:)'
% i=i+7;
% end

% f_re=reshape(y_f',[54,1,7]);
% f_re=reshape(f_re,[54,7]);
% i=1;
% for j=1:length(f_re)
% f_unroll(i:i+7-1,1)=f_re(j,:)'
% i=i+7;
% end
%
% figure,plot(f_unroll,'r'),hold on,plot(t_unroll,'g');
% 
% dltestgen = predict(dlnetGenerator,dlin);
% 
%   y_in={};
%  % o={};
% i_=1;
% r_=1;
% for j_=1:length(feat)
%     if mod(j_,1)==0
%        l_=feat(i_:j_,:);
%        y_in(r_,:)={l_'};
%        i_=i_+1;
%        r_=r_+1;
%     end
% end
% y_in_arr=cell2mat(y_in(1:324))
% y_in_arr_re=reshape(y_in_arr,[324,12,1])
% dlin=dlarray(y_in_arr_re,'BCT')
% 
%         if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%             dlin=gpuArray(dlin);
%         end
%% noise input dlZ
% 
% dlCheck = predict(dlnetGenerator,dlZ);
% % dlCheck and dlX need to be unrolled
% Check_arr=extractdata(dlCheck);
% Check_arr_mat=reshape(Check_arr,[54,7]);
% i=1;
% for j=1:length(Check_arr_mat)
% gan_feat_pred_unrolled_dum(i:i+7-1,1)=xval_arr(j,:)';
% gan_feat_cell_o_test_unrolled_dum(i:i+7-1,1)=Check_arr_mat(j,:)';
% i=i+7;
% end
% figure(23),plot(gan_feat_pred_unrolled_dum,'r'), hold on, plot(gan_feat_cell_o_test_unrolled_dum,'g')