from enum import Enum


class LogTag(Enum):
    """The available logging tags that this evaluation system can use"""
    DateTime = 0,
    AlgorithmType = 1,
    DatasetType = 2,
    RandomElement = 3,
    TargetAccuracy = 4,
    SampleAmount = 5,
    Hyperparameters = 6,
    ConstructionStep = 7,
    ConstructionLoss = 8,
    ConstructionAccuracy = 9,
    ConstructionMSE = 10,
    ConstructionTotalParameters = 11,
    ConstructionPrunedParameters = 12
    ConstructionTrainableParameters = 13,
    ConstructionTime = 14,
    ConstructionStepEpochs = 15,
    TestLoss = 16,
    TestAccuracy = 17,
    TestMSE = 18,
    PruningActive = 19,

    VisualizationPlot = 51,
    ResultHistoryPlot = 52,
    ArchitecturePlot = 53,
