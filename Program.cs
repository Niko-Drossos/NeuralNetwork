/*
* Niko Drossos
* Started: 8/23/24
* 
* I want to try to make a Neural Network from scratch in C#
* so that I can learn about both at the same time.
*/


using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization;
using Newtonsoft.Json;
using static System.Console;
using static NeuralNetwork.Node;

// TODO: Add later when i split into multiple files
// #include "NeuralNetwork.cs";


namespace NeuralNetwork
{
    public class Program
	{
		public static void Main()
		{
			// Network baseNetwork = new();
			// baseNetwork.Generate(2, 3, 4, 2); //(inputs, hidden Layers, nodes per layer, outputs)	
			Network baseNetwork = Utils.Serializer.LoadNetwork("network.json"); 
			baseNetwork.Run();
			baseNetwork.DisplayNodes();

			// This is the value the network wants to output
			float target = 2.384f;
			WriteLine($"Total output: {baseNetwork.GetOutput()}");
			WriteLine($"Target value: {target}");
			WriteLine("------------------------");
			WriteLine($"Difference value: {baseNetwork.TestFitness(target)}");

			// Node testingNode = baseNetwork.Nodes.Values.ElementAt(0);
			Utils.Serializer.SaveNetwork(baseNetwork, "network.json");
			// Network loadedNetwork = Utils.NetworkSerializer.LoadNetwork("network.json");
			// string json = File.ReadAllText("network.json");
			// Network.Layer inputLayer = baseNetwork.Layers[0];
			// Network.Layer hiddenLayer = baseNetwork.Layers[1];
			// testingNode.Activate();

			// testingNode.Mutate();

			// testingNode.DisplayInfo();
			// inputLayer.ActivateNodes();
			// hiddenLayer.ListNodes();
		}
	}

	// Standard Util class for random methods
	public static class Utils
	{
		private static readonly Random random = new();

		public static float InitializeWeightOrBias(float value = 0)
		{
			if (value != 0) 
			{    
				return value;
			}
			else 
			{
				float randomBias = (float)random.NextDouble(); 
				// Give it a 25% chance of being a negative value 
				if ((float)random.NextDouble() > 0.75f) randomBias *= -1;
				return (float)Math.Round(randomBias, 3);
			}
		}

		// Activation function for the nodes 
		public static float LeakyRelu(float value)
		{
			value = (float)Math.Round(value, 3);
			if (value > 0) return value;
			else return (float)(value * 0.1);
		}

		public class Serializer
		{
			public static void SaveNetwork(Network network, string filePath)
			{
				string json = JsonConvert.SerializeObject(network, Formatting.Indented);
				File.WriteAllText(filePath, json);
			}

			public static Network LoadNetwork(string filePath)
			{
				string json = File.ReadAllText(filePath);
				return JsonConvert.DeserializeObject<Network>(json) ?? new Network();
			}
		}
	}

	[DataContract(IsReference = true)]
	public class Network
	{
		[DataMember]
		public Dictionary<string, Node> Nodes { get; set; } = [];
		[DataMember]
		public List<Node> InputNodes { get; set; } = [];
		[DataMember]
		public List<Node> OutputNodes { get; set; } = [];
		[DataMember]
		public List<Layer> Layers { get; set; } = [];

		public void Generate(int inputCount, int layerCount, int nodesPerLayer, int outputCount)
		{	
			// Create a list of input nodes to accept the data input 
			for (int i = 0; i < inputCount; i++)
			{
				float newBias = Utils.InitializeWeightOrBias();
				InputNode inputNode = new(1f, this, newBias);
				InputNodes.Add(inputNode);
				Nodes.Add(inputNode.Id, inputNode);
			}
			
			// The first layer in the network is the input layer
			Layer inputLayer = new(1, InputNodes);  
			Layers.Add(inputLayer);

			// Generate the rest of the hidden layers
			for (int i = 0; i < layerCount; i++)
			{
				GenerateLayer(nodesPerLayer, i);
			}

			// The last layer in the network is the output layer
			for (int i = 0; i < outputCount; i++)
			{
				List<Connection> connectionsToOutput = [];
				OutputNode outputNode = new(connectionsToOutput, this);
				OutputNodes.Add(outputNode);

				foreach (Node node in Layers[layerCount].Nodes)
				{
					Connection toOutputNode = new(Layers[layerCount].Nodes[i].Id, outputNode.Id);
					node.Connect(outputNode);
				}

				Nodes.Add(outputNode.Id, outputNode);
			}

			Layer outputLayer = new(layerCount + 2, OutputNodes);
			Layers.Add(outputLayer);
		}

		public void GenerateLayer(int nodeCount, int layerCount)
		{
			List<Node> layerNodes = [];

			for (int i = 0; i < nodeCount; i++)
			{
				Node newNode = GenerateNode();

				// Connect all nodes in the last layer to all nodes in the next layer
				foreach (Node node in Layers[layerCount].Nodes)
				{
					node.Connect(newNode);
				}

				Nodes.Add(newNode.Id, newNode);
				layerNodes.Add(newNode);
			}

			int layerNumber = Layers.Count + 1;
			Layer newLayer = new(layerNumber, layerNodes);

			Layers.Add(newLayer);
		}

		private Node GenerateNode()
		{
            Node newNode = new([], this);
            return newNode;
		}

		public void Mutate()
		{
			// TODO: Create a method to mutate the network
		}

		public void Run()
		{
			// Run the network
			foreach (Layer layer in Layers)
			{
				layer.ActivateNodes();
			}
		}

		[DataContract(IsReference = true)]
		public class Layer(int layerNumber, List<Node> nodes)
        {
			[DataMember]
            public int LayerNumber { get; set; } = layerNumber;
			[DataMember]
            public List<Node> Nodes { get; set; } = nodes;

            public void ListNodes()
			{	
				WriteLine($"Layer number: {LayerNumber}");
				foreach (Node node in Nodes)
				{
					WriteLine("------------------------");
					node.DisplayInfo();
				}
			}

			public void ActivateNodes()
			{
				foreach (Node node in Nodes)
				{
					node.Activate();
				}
			}
		}


		public void DisplayNodes()
		{
			WriteLine("========================");
			foreach (Layer layer in Layers)
			{
				layer.ListNodes();
				WriteLine("========================");
			}
		}

		public float GetOutput()
		{
			float output = 0.0f;
			// Returns the activation of the output nodes in the network
			foreach (Node node in OutputNodes)
			{
				output += node.Activation;
			}
			return (float)Math.Round(output, 3);
		}

		public float TestFitness(float target)
		{
			float difference = Math.Abs(target - GetOutput());
			return (float)Math.Round(difference, 3);
		}
	}


	// This is just so that you can have inputs into the network
	[DataContract]
	public class InputNode : Node
	{

		public InputNode(float inputValue, Network network, float bias)
			: base(bias, network)
		{
			Activation = inputValue;
			OutConnections = [];
		}

		public float GetOutput()
		{
			return Activation + Bias; // Example output calculation
		}
	}


	// This will contain the resulting outputs in the network
	[DataContract]
	public class OutputNode(List<Connection> inConnections, Network network, float bias = 0.0f) : Node(inConnections, network, bias)
	{
        public float OutputValue { get; private set; } = 0.0f; // Initialize output value to 0
    }
}