import numpy as np


def unbatch_dataloaders_to_numpy(dataloader):
    """
    Unbatch the dataloader to numpy arrays for the input features, labels and engineered features. The engineered features (if they exist) are also unbatched.

    The shape of the input feature is (n_samples, n_timesteps)
    The shape of the labels is (n_samples, 1)
    The shape of the engineered features is (n_featurees, n_samples, n_timesteps)

    :param dataloader: The dataloader to unbatch
    :return: The unbatched input features, labels and engineered features as numpy arrays


    """
    unbatched_input_feature = []
    unbatched_labels = []
    unbatched_eng_features = []
    for i, batch in enumerate(dataloader):
        features, labels = batch
        features_array, labels_array = features.numpy(), labels.numpy()
        labels_array = labels_array.reshape(-1, 1)
        _, _, no_of_features = features_array.shape

        # Iterate through each feature
        for j in range(no_of_features):
            feature_data = features_array[:, :, j]

            # This only happens during the first batch
            if len(unbatched_eng_features) == j:
                unbatched_eng_features.append(feature_data)
            elif len(unbatched_eng_features) < j:
                raise ValueError(
                    "The length of the unbatched_eng_features array is less than the index which should not happen"
                )
            # This happens for all subsequent batches
            else:
                unbatched_eng_features[j] = np.concatenate(
                    (unbatched_eng_features[j], feature_data), axis=0
                )

        # Append the feature data to the unbatched_input_feature list
        unbatched_input_feature.append(features_array[:, :, 0])
        # Append the labels to the unbatched_labels list
        unbatched_labels.append(labels_array)

    # Concatenate the unbatched_input_feature array (along the first axis to create a single array)
    unbatched_input_feature = np.concatenate(unbatched_input_feature, axis=0)
    # Concatenate the unbatched_labels array (along the first axis to create a single array)
    unbatched_labels = np.concatenate(unbatched_labels, axis=0)
    # Concatenate the unbatched_eng_features array (no concatenation is required as it is already a single array)
    unbatched_eng_features = np.array(unbatched_eng_features)

    return unbatched_input_feature, unbatched_labels, unbatched_eng_features
