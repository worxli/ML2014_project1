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
lambda = 0.1:0.2:20;

%kfold default=10
kfold = 10;

maxNumFeatures = 5;

selectedFeatures = [];

% Loop over feature space until enough found
while length(selectedFeatures) < maxNumFeatures
    
    featureErrors = [];
    
    % For each feature compute the error for a random for a certain labmda
    for featIdx=1:size(Xt,2)

        % For all lambdas compute crossvalidation
        [errs, index] = crossValidation(Xt, y, lambda, kfold);
        
        % Store all errors for one feature
        featureErrors = [featureErrors min(errs)];
        
    end

    % Choose the feature with the smallest error
    [minFeatureError, idx] = min(featureErrors);

    % Store index of the selected feature
    selectedFeatures = [selectedFeatures idx];

end

% TODO: Xt(:,selectedFeatures) Run cross validation with all features
% And find best lambda
[errs, index] = crossValidation(Xt(:,selectedFeatures),y,lambda,kfold);

% Plot lambda error for the final selected feature set
plot(lambda,errs);

%get index for lambda with lowest error
[val, ind] = min(errs);
disp(['Prediction error for lambda ' num2str(lambda(ind)) ' is: ' num2str(val) ' (chosen lambda), MODEL ERROR: ' num2str(sum(errs))]);

%calculate beta with chosen lambda
ridgebeta = regression(Xt,y,lambda(ind));

%calculate and show error for beta estimate
ridgeerr = Xt*ridgebeta-y;
disp(['Error on training data: ' num2str(norm(ridgeerr)) ]);

%% test on validation set

% normalize validation data
averagedata = validation-repmat(MEAN(1:end-1),size(validation,1),1);
normdata = bsxfun(@rdivide, averagedata, STD(1:end-1));

%model definition
normdata = getFeatures(normdata);
normdata = normdata(:, Bidx);

% calculate prediction and un-normalize
prediction = normdata*ridgebeta;
unnormpred = bsxfun(@times, prediction, STD(end));
preddata = unnormpred+repmat(MEAN(end),size(validation,1),1);

%% write to csv file for submission
csvwrite('validationsetresult.csv', preddata);







