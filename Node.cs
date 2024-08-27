/*
* Niko Drossos
* Started: 8/23/24
* 
* I want to try to make a Neural Network from scratch in C#
* so that I can learn about both at the same time.
*/

using Newtonsoft.Json;
using System.Runtime.Serialization;
using static System.Console;

// TODO: Add later when i split into multiple files
// #include "NeuralNetwork.cs";


namespace NeuralNetwork
{
    // This is for the individual Nodes in the network
    [DataContract(IsReference = true)]
	[KnownType(typeof(Connection))]
	[KnownType(typeof(Node))]
	[KnownType(typeof(InputNode))]
	[KnownType(typeof(OutputNode))]
	public class Node
	{
		[DataMember]
		public string Id { get; }
		[DataMember]
		public float Bias { get; set; }
		[DataMember]
		public float Activation { get; set; }
		[DataMember]
		public List<Connection> OutConnections { get; set; }
		[DataMember]
		public List<Connection> InConnections { get; set; }
		[DataMember]
		public Network Network { get; set; }

		[DataContract]
		public class Connection(string FromNode, string ToNode, float Weight = 0.0f)
		{
			[DataMember]
			public float Weight { get; set; } = Utils.InitializeWeightOrBias(Weight);
			[DataMember]
			public string FromNode { get; set; } = FromNode;
			[DataMember]
			public string ToNode { get; set; } = ToNode;

			public void Info()
			{
				WriteLine($"weight {Weight}, from {FromNode}, to {ToNode}");
			}
		}

		[JsonConstructor]
		public Node(List<Connection> inConnections, Network network, float bias = 0.0f)
		{
			Random random = new();
			Id = Convert.ToString(random.Next());
			Bias = Utils.InitializeWeightOrBias(bias);
			OutConnections = [];
			InConnections = inConnections;
			Network = network;
			Activation = 0.0f;
		}

		public Node(float bias, Network network)
		{
			Random random = new();
			Id = Convert.ToString(random.Next());
			Bias = bias;
			OutConnections = [];
			InConnections = [];
			Network = network;
			Activation = 0.0f;
		}


		// Just displays the information stored inside the Node
		public void DisplayInfo()
		{
			WriteLine($"Id: {Id}");
			WriteLine($"Bias: {Bias}");
			WriteLine("InConnection:");
			foreach (Connection connection in InConnections)
			{
				connection.Info();
			}
			WriteLine("OutConnections:");
			foreach (Connection connection in OutConnections)
			{
				connection.Info();
			}
			WriteLine($"Activation: {Activation}");
		}

		// Connects the Node to another, outNode.Connect(inNode)
		public void Connect(Node node, float bias = 0.0f)
		{
			float connectionWeight = Utils.InitializeWeightOrBias(bias);

			Connection nodeEdge = new(Id, node.Id, connectionWeight);
			
			// Add the edge to the inConnections of the node it connects
			node.InConnections.Add(nodeEdge);

			// Add the edge to the outConnections of the node
			OutConnections.Add(nodeEdge);
		}

		// Send the data from th node to the next layer
		public void Activate()
		{
			float newOutput = 0.0f;
			// Take the sum of all the inputs in the previous layer
			// And multiply it by the weight on that connection
			foreach (Connection connection in InConnections)
			{
				Node fromNode = Network.Nodes[connection.FromNode];
				newOutput += connection.Weight * fromNode.Activation; 
			}

			Activation = Utils.LeakyRelu((float)newOutput + Bias);
		}

		public void Mutate()
		{
			
			// Choose a random number between 0 and 2.
			// This is used to choose where he mutation takes place
			
			Random random = new();
			byte randomByte = (byte)random.Next(0, 2);
			
			if (randomByte == 0)
			{
				float newBias = Utils.InitializeWeightOrBias();
				WriteLine($"Mutating bias on node {Id} from {Bias} to {newBias}");
				Bias = newBias;
			}
			else if (randomByte == 1)
			{
				byte randomEdge = (byte)random.Next(0, OutConnections.Count);
				float newWeight = Utils.InitializeWeightOrBias();
				WriteLine($"Mutating weight on node {Id} from {OutConnections[randomEdge].Weight} to {newWeight}");
				OutConnections[randomEdge].Weight = newWeight;
			}
		}
	}
}