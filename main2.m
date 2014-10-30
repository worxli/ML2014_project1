clear;

%% read data
training = csvread('training.csv');
%testdata = csvread('testing.csv');
validation = csvread('validation.csv');

%%  normalization

MEAN = mean(training);
STD = std(training);
averagedata = training-repmat(MEAN,size(training,1),1);
normdata = bsxfun(@rdivide, averagedata, STD);

%Features
X = normdata(:,1:end-1);
y = normdata(:,end);
Xt = getFeatures(X);

%% Maybe use sequentials

%% ridge regression

%possible lambdas
%lambda = exp(-1:0.1:5);
lambda = 1:1:10;

%kfold default=10
kfold = 5;

%maxNumFeatures = 5;

%selectedFeatures = [];
% 
% % Loop over feature until enough found
% 
% while length(selectedFeatures) < maxNumFeatures
%     
%     featureErrors = [];
%     
%     % Iterate over feature space for feature selection
%     for featIdx=1:size(Xt,2)
% 
%         [errs, index] = crossValidation(Xt, y, lambda, kfold);
%         
%         featureErrors = [featureErrors min(errs)];
%     end
% 
%     [minFeatureError, idx] = min(featureErrors);
% 
%     selectedFeatures = [selectedFeatures idx];
% 
% end

featurevec = 1:size(Xt,2);
errs = [];
indices = [];
featuresvec = [];
for i = 2:size(Xt,2)
   
    features = nchoosek(featurevec,i);
    if size(features,2)>1
        for j = 1:size(features,1)
            [err, index] = crossValidation(Xt(:,features(j,:)),y,lambda,kfold);
            errs = [errs err];
            indices = [indices index];
            emptyvec = zeros(length(featurevec) ,1);
            emptyvec(1:length(features(j,:))) = features(j,:);
            featuresvec = [featuresvec; emptyvec'];
        end
    else
        [err, index] = crossValidation(Xt(:,features),y,lambda,kfold); 
        errs = [errs err];
        indices = [indices index];
        emptyvec = zeros(length(featurevec) ,1);
        emptyvec(1:length(features)) = features;
        featuresvec = [featuresvec; emptyvec'];
    end
    
end



plot(errs);

%get index for lambda with lowest error
[val, ind] = min(errs);
disp(['Prediction error for lambda ' num2str(lambda(ind)) ' is: ' num2str(val) ' (chosen lambda), MODEL ERROR: ' num2str(sum(errs))]);
featuresvc(ind,:)

%calculate beta with chosen lambda
ridgebeta = regression(Xt(:,selectedFeatures),y,lambda(ind));

%calculate and show error for beta estimate
ridgeerr = Xt*ridgebeta-y;
disp(['Error on training data: ' num2str(norm(ridgeerr)) ]);

%% test on validation set

% normalize validation data
averagedata = validation-repmat(MEAN(1:end-1),size(validation,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

%model definition
normdata = getFeatures(normdata);
normdata = normdata(:, selectedFeatures);

% calculate prediction and un-normalize
prediction = normdata*ridgebeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(validation,1),1);

%% write to csv file for submission
csvwrite('validationsetresult.csv', preddata);







