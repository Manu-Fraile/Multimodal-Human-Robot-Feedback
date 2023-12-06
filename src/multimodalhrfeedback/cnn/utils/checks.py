import pandas as pd
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.utils.validation.panel import check_X, check_X_y

from inspect import isclass
from pathlib import Path

from sktime.exceptions import NotFittedError


def check_and_clean_data(X, y=None, input_checks=True):
    '''
    Performs basic sktime data checks and prepares the train data for input to
    Keras models.
    Parameters
    ----------
    X: the train data
    y: the train labels
    input_checks: whether to perform the basic sktime checks
    Returns
    -------
    X
    '''
    if input_checks:
        if y is None:
            check_X(X)
        else:
            check_X_y(X, y)

    # want data in form: [instances = n][timepoints = m][dimensions = d]
    if isinstance(X, pd.DataFrame):
        if _is_nested_dataframe(X):
            if X.shape[1] > 1:
                # we have multiple columns, AND each cell contains a series,
                # so this is a multidimensional problem
                X = _multivariate_nested_df_to_array(X)
            else:
                # we have a single column containing a series, treat this as
                # a univariate problem
                X = _univariate_nested_df_to_array(X)
        else:
            # we have multiple columns each containing a primitive, treat as
            # univariate series
            X = _univariate_df_to_array(X)

    if len(X.shape) == 2:
        # add a dimension to make it multivariate with one dimension
        X = X.values.reshape(
            X.shape[0], X.shape[1], 1
        )  # go from [n][m] to [n][m][d=1]
    # return transposed data to conform with current model formats
    return X.transpose(0, 2, 1)


def check_and_clean_validation_data(validation_X, validation_y,
                                    label_encoder=None,
                                    onehot_encoder=None, input_checks=True):
    '''
    Performs basic sktime data checks and prepares the validation data for
    input to Keras models. Also provides functionality to encode the y labels
    using label encoders that should have already been fit to the train data.
    :param validation_X: required, validation data
    :param validation_y: optional, y labels for categorical conversion if
            needed
    :param label_encoder: if validation_y has been given,
            the encoder that has already been fit to the train data
    :param onehot_encoder: if validation_y has been given,
            the encoder that has already been fit to the train data
    :param input_checks: whether to perform the basic input structure checks
    :return: ( validation_X, validation_y ), or None if no data given
    '''
    if validation_X is not None:
        validation_X = check_and_clean_data(validation_X, validation_y,
                                            input_checks=input_checks)
    else:
        return None

    if label_encoder is not None and onehot_encoder is not None:
        validation_y = label_encoder.transform(validation_y)
        validation_y = validation_y.reshape(
            len(validation_y), 1)
        validation_y = onehot_encoder.fit_transform(
            validation_y)

    return (validation_X, validation_y)


def check_is_fitted(estimator, msg=None):
    """Perform is_fitted validation for estimator.
    Adapted from sklearn.utils.validation.check_is_fitted
    Checks if the estimator is fitted by verifying the presence and
    positivity of self.is_fitted_
    Parameters
    ----------
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    Returns
    -------
    None
    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not hasattr(estimator, "_is_fitted") or not estimator.is_fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def save_trained_model(
        model, model_save_directory, model_name, save_format="h5"
):
    """
    Saves the model to an HDF file.
    Saved models can be reinstantiated via `keras.models.load_model`.
    Parameters
    ----------
    save_format: string
        'h5'. Defaults to 'h5' currently but future releases
        will default to 'tf', the TensorFlow SavedModel format.
    """
    if save_format != "h5":
        raise ValueError(
            "save_format must be 'h5'. This is the only format "
            "currently supported."
        )
    if model_save_directory is not None:
        if model_name is None:
            file_name = "trained_model.hdf5"
        else:
            file_name = model_name + ".hdf5"
        path = Path(model_save_directory) / file_name
        model.save(
            path
        )  # Add save_format here upon migration from keras to tf.keras