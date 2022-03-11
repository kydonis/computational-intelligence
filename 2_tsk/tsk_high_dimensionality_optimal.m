clc
clear all

data = table2array( readtable('train.csv', 'HeaderLines',1));

new_data = data(randperm(size(data,1)),:);

Dtrn = new_data(1:floor(size(new_data,1)*0.8),:);
Dchk = new_data(size(Dtrn,1)+1:end, :);

[Dtrnx,PS1] = mapminmax(Dtrn(:,1:end-1)',0,1);
Dtrnx = Dtrnx';
Dchkx = mapminmax('apply',Dchk(:,1:end-1)',PS1);
Dchkx= Dchkx';

[Dtrny,PS2] = mapminmax(Dtrn(:,end)',0,1);
Dtrny = Dtrny';
Dchky = mapminmax('apply',Dchk(:,end)',PS2);
Dchky= Dchky';

% parameters
features = 4;
radius = 0.1;

indexes=[63 66 31 1];

genfis_opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',radius);
new_fis = genfis(Dtrnx(:,indexes(1:features)),Dtrny(:),genfis_opt);

trn_options = anfisOptions('InitialFis',new_fis,'EpochNumber',100);
[trnFis,trnError,stepSize] = anfis([Dtrnx(:,indexes(1:features)) Dtrny(:)],trn_options);

y_out = evalfis(trnFis, Dchkx(:,indexes(1:features))); 
pred_error = Dchk(:,end) - mapminmax('reverse',y_out',PS2)'; 

figure
plot(trnError, 'LineWidth',2);
xlabel('Epochs');
ylabel('Error');
name = strcat('optimal_tsk_epochs_error');
saveas(gcf,name,'png');

figure;
plot([Dchk(:,end) mapminmax('reverse',y_out',PS2)'], 'LineWidth',2);
xlabel('True');
ylabel('Predicted');
name = strcat('optimal_tsk_true_predicted');
saveas(gcf,name,'png');

figure;
plot(pred_error, 'LineWidth',2);
xlabel('input');
ylabel('Error');
title(strcat("Optimal Tsk model ", strcat("Prediction Error")));
name = strcat('optimal_tsk_model_prediction_error');
saveas(gcf,name,'png');

y_out2 = mapminmax('reverse',y_out',PS2)';    
SSres = sum((Dchk(:,end) - y_out2).^2);
SStot = sum((Dchk(:,end) - mean(Dchk(:,end))).^2);
R2 = 1- SSres/SStot;
NMSE = 1-R2;
RMSE = sqrt(mse(y_out2,Dchk(:,end)));
NDEI = sqrt(NMSE);

figure;
plot(Dchk(:,end), y_out2, '.');
xlabel('True');
ylabel('Predicted');
name = strcat('optimal_tsk_model_true_predicted_2');
saveas(gcf,name,'png');

fileID = fopen('optimal_results.txt','w');
fprintf(fileID,'TSK_model_optimal');
fprintf(fileID,'\nRMSE = %f\n NMSE = %f\n NDEI = %f\n R2 = %f\n\n', RMSE, NMSE, NDEI, R2);
fclose(fileID);