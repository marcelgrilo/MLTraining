using System.IO.Pipes;
using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace MultiClassClassification
{
    public class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {

            _mlContext = new MLContext(seed: 0);

            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath,hasHeader: true);

            var pipeline = ProcessData();

            // constroi e treina o pipeline.
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);


            Evaluate(_trainingDataView.Schema);

            PredictIssue();


        }

        /// <summary>
        /// extração de caracteristicas e transformação dos dados.
        /// </summary>
        /// <returns></returns>
        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion
                // mapeia a area para valores numericos
                .MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                // transforma o titulo e a descricao para vetores numericos
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    outputColumnName: "TitleFeaturized",
                    inputColumnName: "Title"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    outputColumnName: "DescriptionFeaturized",
                    inputColumnName: "Description"))
                // combina as features criadas acime em apenas 1 vetor numerico de coluna chamada Features
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                // melhora de performance
                .AppendCacheCheckpoint(_mlContext);
            return pipeline;
        }

        /// <summary>
        /// cria e freina o modelo, retonando um motor de predição para od dados
        /// </summary>
        /// <param name="trainingDataView"></param>
        /// <param name="pipeline"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildAndTrainModel(
            IDataView trainingDataView,
            IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline
                //escolha do algoritmo de classificaçã para multiclasse
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // o treino do modelo
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            // cria o motor de predição
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            // faz um pequeno teste de predição
            GitHubIssue issue = new GitHubIssue() {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };
            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            return trainingPipeline;
        }

        /// <summary>
        /// avalia o modelo de dados e se a assertividade do motor de predição esta bom
        /// </summary>
        /// <param name="trainingDataViewSchema"></param>
        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath,hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        private static void SaveModelAsFile(MLContext mlContext,DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        private static void PredictIssue()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            GitHubIssue singleIssue = new GitHubIssue() {
                Title = "Entity Framework crashes",
                Description = "When connecting to the database, EF is crashing"
            };
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
            var prediction = _predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");

        }
    }

}