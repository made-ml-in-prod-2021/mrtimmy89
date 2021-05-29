from sklearn.ensemble import RandomForestClassifier

from src.data.make_dataset import read_data, dataset_split
from src.features.make_features import extract_target, full_transform
from src.models.fit_predict_model import predict_model, train_model
from src.entities.split_parameters import SplittingParams


def test_train_model(dataset_path: str) -> None:
    df = read_data(dataset_path)
    target = extract_target(df)
    splitting_params = SplittingParams()
    df_transformed = full_transform(df)
    X_train, X_test, y_train, y_test = dataset_split(
        df_transformed,
        target,
        splitting_params
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_predict_model(dataset_path: str) -> None:
    df = read_data(dataset_path)
    target = extract_target(df)
    splitting_params = SplittingParams()
    df_transformed = full_transform(df)
    X_train, X_test, y_train, y_test = dataset_split(
        df_transformed,
        target,
        splitting_params
    )
    model = train_model(df_train=X_train, target=y_train)
    pred_labels, pred_proba = predict_model(model, X_test)
    assert len(set(pred_labels)) == 2
    assert max(pred_proba) < 1
