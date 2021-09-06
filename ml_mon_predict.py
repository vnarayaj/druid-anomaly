import logging
import json
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.logging import TimeElapsedLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from kafka import KafkaProducer
from pykafka import KafkaClient
from datetime import datetime

client = KafkaClient(hosts='localhost:9092')
topic = client.topics['ml1']
producer = topic.get_sync_producer()
logging.basicConfig(level=logging.INFO)

X, y = load_iris(return_X_y=True)
rf = RandomForestClassifier()

def pipeline():
    """A dummy model that has a bunch of components that we can test."""
    r=random.randrange(1,5)
    if r==4:
        nest=100
    else:
        nest=10
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("normal", Normalizer()),
            (
                "union",
                FeatureUnion(
                    [
                        ("pca", PCA(n_components=1)),
                        ("svd", TruncatedSVD(n_components=2)),
                    ],
                    n_jobs=1,  # parallelized components won't generate spans
                ),
            ),
            ("class", RandomForestClassifier(n_estimators=nest)),
        ]
    )
    X_train,y_train=load_iris(return_X_y=True)
    model.fit(X_train, y_train)
    return model
def random_input():
    """A random record from the feature set."""
    rows = X.shape[0]
    random_row = np.random.choice(rows, size=1)
    return X[random_row, :]


rf.fit(X, y)
rf.predict(X)

model = pipeline()
#instrumentor3 = SklearnInstrumentor(instrument=OpenTelemetrySpanner())
#instrumentor3.instrument_estimator(model)
j=1
while j:
    r=random.randrange(0,100)
    x_test = random_input()+1
    #x_test[0,1]=x_test[0,1]+r
    x1=x_test.tolist()
    z=model.predict(x_test)
    z1=z.tolist()
    m={}
    t=datetime.now()
    y=t.strftime("%Y-%m-%d %H:%M:%S.%f")
    m={"ts":y,"col1":x1[0][0],"col2":x1[0][1],"col3":x1[0][2],"col4":x1[0][3],"predict":z1[0]}
    n=json.dumps(m)
    producer.produce(n.encode('ascii'))
    #print(n)
# No more logging
#rf.predict(X)
