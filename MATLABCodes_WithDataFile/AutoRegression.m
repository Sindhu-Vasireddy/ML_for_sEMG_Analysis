%     ze=iddata(double(feat_out(1:1350,:)),double(feat(1:1350,:)),1/100);%70% train
%     zv=iddata(double(feat_out(1351:end,:)),double(feat(1351:end,:)),1/100);%30% test
%% Linear arx
%12i
Di_12i_1o=iddata(double(feat_out(:,4)),double(feat),1/100);
[sys_12i_1o_l,fit_Bay_12i_1o_l,RMSE_arx_12i_1o_l]=linear_arx(Di_12i_1o)
display("Ran 12i 1o linear");

Di_12i_2o=iddata(double(feat_out(:,3:4)),double(feat),1/100);
[sys_12i_2o,fit_Bay_12i_2o_l,RMSE_arx_12i_2o_l]=linear_arx(Di_12i_2o)
display("Ran 12i 2o linear");

%10i
Di_10i_1o=iddata(double(feat_out(:,4)),double([feat(:,1:8) feat(:,11:12)]),1/100);
[sys_10i_1o_l,fit_Bay_10i_1o_l,RMSE_arx_10i_1o_l]=linear_arx(Di_10i_1o)
display("Ran 10i 1o linear");

Di_10i_2o=iddata(double(feat_out(:,3:4)),double([feat(:,1:8) feat(:,11:12)]),1/100);
[sys_10i_2o,fit_Bay_10i_2o_l,RMSE_arx_10i_2o_l]=linear_arx(Di_10i_2o)
display("Ran 10i 2o linear");

%2i
ze=iddata(double(feat_out(1:1350,4)),double(feat(1:1350,11:12)),1/100);%70% train
zv=iddata(double(feat_out(1351:end,4)),double(feat(1351:end,11:12)),1/100);%30% test
Di_2i_1o=iddata(double(feat_out(:,4)),double(feat(:,11:12)),1/100);
[sys_2i_1o_l,fit_Bay_2i_1o_l,RMSE_arx_2i_1o_l]=linear_arx(Di_2i_1o,ze,zv)
display("Ran 2i 1o linear");

ze=iddata(double(feat_out(1:1350,3:4)),double(feat(1:1350,11:12)),1/100);%70% train
zv=iddata(double(feat_out(1351:end,3:4)),double(feat(1351:end,11:12)),1/100);%30% test
Di_2i_2o=iddata(double(feat_out(:,3:4)),double(feat(:,11:12)),1/100);
[sys_2i_2o,fit_Bay_2i_2o_l,RMSE_arx_2i_2o_l]=linear_arx(Di_2i_2o,ze,zv)
display("Ran 2i 2o linear");



%% Non-Linear arx
%12i
[sys_12i_1o_nl,fit_Bay_12i_1o_nl,RMSE_nlarx_12i_1o_nl]=nonlinear_arx(Di_12i_1o) 
display("Ran 12i 1o non-linear");
[sys_12i_2o_nl,fit_Bay_12i_2o_nl,RMSE_nlarx_12i_2o_nl]=nonlinear_arx(Di_12i_2o)
display("Ran 12i 2o non-linear");

%10i
[sys_10i_1o_nl,fit_Bay_10i_1o_nl,RMSE_nlarx_10i_1o_nl]=nonlinear_arx(Di_10i_1o) 
display("Ran 10i 1o non-linear");
[sys_10i_2o_nl,fit_Bay_10i_2o_nl,RMSE_nlarx_10i_2o_nl]=nonlinear_arx(Di_10i_2o)
display("Ran 10i 2o non-linear");

%2i
ze=iddata(double(feat_out(1:1350,4)),double(feat(1:1350,11:12)),1/100);%70% train
zv=iddata(double(feat_out(1351:end,4)),double(feat(1351:end,11:12)),1/100);%30% test
[sys_2i_1o_nl,fit_Bay_2i_1o_nl,RMSE_nlarx_2i_1o_nl]=nonlinear_arx(Di_2i_1o,ze,zv) 
display("Ran 2i 1o non-linear");
ze=iddata(double(feat_out(1:1350,3:4)),double(feat(1:1350,11:12)),1/100);%70% train
zv=iddata(double(feat_out(1351:end,3:4)),double(feat(1351:end,11:12)),1/100);%30% test
[sys_2i_2o_nl,fit_Bay_2i_2o_nl,RMSE_nlarx_2i_2o_nl]=nonlinear_arx(Di_2i_2o,ze,zv)
display("Ran 2i 2o non-linear");
