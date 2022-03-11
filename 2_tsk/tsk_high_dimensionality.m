clc
clear all

data = load('airfoil_self_noise.dat');

new_data = data(randperm(size(data,1)),:);

Dtrn = new_data(1:floor(size(new_data,1)*0.6),:);
Dval = new_data(size(Dtrn,1)+1:size(Dtrn,1)+ceil(size(new_data,1)*0.2),:);
Dchk = new_data(size(Dtrn,1)+size(Dval,1)+1:end, :);

[Dtrn,PS] = mapminmax(Dtrn',0,1);
Dtrn = Dtrn';
Dval = mapminmax('apply',Dval',PS);
Dval= Dval';
Dchk = mapminmax('apply',Dchk',PS);
Dchk= Dchk';

opt(1) = genfisOptions('GridPartition');
opt(1).InputMembershipFunctionType = 'gbellmf';
opt(2) = genfisOptions('GridPartition');
opt(2).InputMembershipFunctionType = 'gbellmf';
opt(3) = genfisOptions('GridPartition');
opt(3).InputMembershipFunctionType = 'gbellmf';
opt(4) = genfisOptions('GridPartition');
opt(4).InputMembershipFunctionType = 'gbellmf';

opt(1).NumMembershipFunctions = 2;
opt(1).OutputMembershipFunctionType = 'constant';
opt(2).NumMembershipFunctions = 3;
opt(2).OutputMembershipFunctionType = 'constant';
opt(3).NumMembershipFunctions = 2;
opt(3).OutputMembershipFunctionType = 'linear';
opt(4).NumMembershipFunctions = 3;
opt(4).OutputMembershipFunctionType = 'linear';
fileID = fopen('results.txt','w');
for i = 1:4
    Tsk_model(i) = genfis(Dtrn(:,1:end-1),Dtrn(:,end),opt(i));
    trn_options = anfisOptions('InitialFis',Tsk_model(i),'EpochNumber',120);
    trn_options.ValidationData = [Dval(:,1:end-1) Dval(:,end)];
    [trnFis,trnError,stepSize,valFis,valError] = anfis([Dtrn(:,1:end-1) Dtrn(:,end)],trn_options);
    
    figure('Renderer', 'painters', 'Position', [10 10 1200 800], 'Name', strcat('TSK model ', int2str(i)));
    sgtitle(strcat('TSK model ', int2str(i)));
    subplot(2,5,1)
    plotmf(Tsk_model(i),'input',1);
    xlabel('Frequency')
    
    subplot(2,5,2)
    plotmf(Tsk_model(i),'input',2);
    xlabel('Angle of attack')
    
    subplot(2,5,3)
    plotmf(Tsk_model(i),'input',3);
    xlabel('Chord length')
    subtitle('Before training');
    
    subplot(2,5,4)
    plotmf(Tsk_model(i),'input',4);
    xlabel('Free-stream velocity')
    
    subplot(2,5,5)
    plotmf(Tsk_model(i),'input',5);
    xlabel('Suction side displacement thickness')
    
    subplot(2,5,6)
    plotmf(valFis,'input',1);
    xlabel('Frequency')
    
    subplot(2,5,7)
    plotmf(valFis,'input',2);
    xlabel('Angle of attack')
    
    subplot(2,5,8)
    plotmf(valFis,'input',3);
    xlabel('Chord length')
    subtitle('After training');
    
    subplot(2,5,9)
    plotmf(valFis,'input',4);
    xlabel('Free-stream velocity')
    
    subplot(2,5,10)
    plotmf(valFis,'input',5);
    xlabel('Suction side displacement thickness')
    name =  strcat('tsk_model_',int2str(i), '_MFs');
    saveas(gcf,name,'png'); 
    
    y_out = evalfis(valFis, Dchk(:,1:end-1)); 
    SSres = sum((Dchk(:,end) - y_out).^2);
    SStot = sum((Dchk(:,end) - mean(Dchk(:,end))).^2);
    R2 = 1- SSres/SStot;
    NMSE = 1-R2;
    RMSE = sqrt(mse(y_out,Dchk(:,end)));
    NDEI = sqrt(NMSE);
    pred_error = Dchk(:,end) - y_out;
    fileID = fopen('results.txt','a');
    fprintf(fileID,strcat('TSK_model_',int2str(i)));
    fprintf(fileID,'\nRMSE = %f\n NMSE = %f\n NDEI = %f\n R2 = %f\n\n', RMSE, NMSE, NDEI, R2);
    
    figure;
    plot([trnError valError], 'LineWidth',2);
    xlabel('Epochs');
    ylabel('Error');
    legend('Training Error', 'Validation Error');
    title(strcat('Tsk model ', strcat(int2str(i), ' Learning Curve')));
    name = strcat('tsk_model_',int2str(i), '_learning_curve');
    saveas(gcf,name,'png'); 

    figure;
    plot(pred_error, 'LineWidth',2);
    xlabel('Testing Data');
    ylabel('Error');
    title(strcat('Tsk model ', strcat(int2str(i), ' Prediction Error')));
    name = strcat('tsk_model_',int2str(i), '_prediction_error');
    saveas(gcf,name,'png'); 

    figure, errorbar(y_out, y_out, min(-pred_error,0),max(-pred_error,0))
            xlabel('True output');
    ylabel('Predicted');
    title(strcat('Tsk model ', strcat(int2str(i), ' with error bars')));
    name = strcat('tsk_model_',int2str(i), '_with_error_bars');
    saveas(gcf,name,'png'); 
    
    figure, hist(pred_error,50)
    title(strcat('Tsk model ', strcat(int2str(i), ' histogram of the prediction errors')));
    name = strcat('tsk_model_',int2str(i), '_histogram_of_the_prediction_errors');
    saveas(gcf,name,'png'); 
end
fclose(fileID);
