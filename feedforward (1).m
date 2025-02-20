%% 
%% Load the Cleaned Dataset
filename = 'thyroidDF_cleaned.csv'; % Adjust the path if needed
data = readtable(filename);

% Separate features and target
X = data{:, 1:end-1}; % All columns except the last (features)
Y = categorical(data.target); % Convert the target column to categorical

%% Split the Data into Training and Validation Sets
rng(42); % Set random seed for reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.4); % 90-10 train-validation split
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_val = X(test(cv), :);
Y_val = Y(test(cv), :);

%% Define Neural Network Architecture
layers = [
    featureInputLayer(size(X_train, 2), 'Normalization', 'none')
    fullyConnectedLayer(14, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(14, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(10, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(numel(categories(Y_train)), 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Define Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...         % Learning rate
    'MaxEpochs', 100, ...                  % Number of epochs
    'MiniBatchSize', 32, ...               % Batch size
    'ValidationData', {X_val, Y_val}, ...  % Validation set
    'ValidationFrequency', 10, ...         % Validation frequency
    'Verbose', false, ...
    'Shuffle', 'every-epoch');             % Shuffle data every epoch

%% Initialize Early Stopping Parameters
monitor_metric = 'accuracy';   % Metric to monitor (accuracy or loss)
patience = 10;                 % Number of epochs to wait for improvement
min_delta = 0.00001;           % Minimum change to qualify as an improvement
restore_best_weights = true;   % Restore the best model
verbose = true;                % Display progress

best_metric = -inf;            % Initialize the best metric
wait = 0;                      % Counter for epochs without improvement
best_weights = [];             % Variable to store the best weights

%% Prepare for Logging Metrics
epoch_training_accuracy = [];  % To store training accuracy
epoch_validation_accuracy = []; % To store validation accuracy
epoch_training_loss = [];      % To store training loss
epoch_validation_loss = [];    % To store validation loss

%% Train the Neural Network with Early Stopping and Logging
stopped_epoch = options.MaxEpochs; % Track when training stopped

for epoch = 1:options.MaxEpochs
    % Train for one epoch
    [trainedNet, trainInfo] = trainNetwork(X_train, Y_train, layers, options);

    % Log metrics
    training_accuracy = trainInfo.TrainingAccuracy(end); % Accuracy of the last epoch
    val_accuracy = trainInfo.ValidationAccuracy(end);    % Validation accuracy of the last epoch
    training_loss = trainInfo.TrainingLoss(end);         % Loss of the last epoch
    val_loss = trainInfo.ValidationLoss(end);            % Validation loss of the last epoch
    
    % Store metrics
    epoch_training_accuracy = [epoch_training_accuracy, training_accuracy];
    epoch_validation_accuracy = [epoch_validation_accuracy, val_accuracy];
    epoch_training_loss = [epoch_training_loss, training_loss];
    epoch_validation_loss = [epoch_validation_loss, val_loss];
    
    % Display training progress
    fprintf('Epoch %d - Training Accuracy: %.4f, Validation Accuracy: %.4f, Training Loss: %.4f, Validation Loss: %.4f\n', ...
        epoch, training_accuracy, val_accuracy, training_loss, val_loss);
    
    % Check for improvement
    if val_accuracy > best_metric + min_delta
        best_metric = val_accuracy;
        wait = 0;
        if restore_best_weights
            best_weights = trainedNet.Layers; % Save best weights
        end
    else
        wait = wait + 1;
    end
    
    % Check for early stopping
    if wait >= patience
        if verbose
            fprintf('Stopping early at epoch %d. Best validation accuracy: %.4f\n', epoch, best_metric);
        end
        stopped_epoch = epoch;
        break;
    end
end

% Restore best weights if applicable
if restore_best_weights && ~isempty(best_weights)
    trainedNet = assembleNetwork(best_weights); % Restore weights
end

%% Plot Training and Validation Metrics
epochs = 1:stopped_epoch;

% Plot Accuracy
figure;
plot(epochs, epoch_training_accuracy(1:stopped_epoch), 'b-', 'LineWidth', 2, 'DisplayName', 'Training Accuracy');
hold on;
plot(epochs, epoch_validation_accuracy(1:stopped_epoch), 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Accuracy');
hold off;
xlabel('Epoch');
ylabel('Accuracy');
title('Training and Validation Accuracy Over Epochs');
legend('show');
grid on;

% Plot Loss
figure;
plot(epochs, epoch_training_loss(1:stopped_epoch), 'b-', 'LineWidth', 2, 'DisplayName', 'Training Loss');
hold on;
plot(epochs, epoch_validation_loss(1:stopped_epoch), 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Loss');
hold off;
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss Over Epochs');
legend('show');
grid on;
