package com.pskoczek.cnn3d;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.dataset.DataSet;

public class Runner {
	private static final int HEIGHT = 96, WIDTH = 96, CHANNELS = 1, DEPTH = 16, OUTPUT_LABELS = 2, MINIBATCHES = 2;

	public static void main(String[] args) {
		RandomDataSetProvider randomProvider = new RandomDataSetProvider(HEIGHT, WIDTH, CHANNELS, DEPTH, MINIBATCHES,
				OUTPUT_LABELS);

		MultiLayerConfiguration configuration = NetworkUtils.createMultilayerConfiguration(DEPTH, HEIGHT, WIDTH,
				CHANNELS, OUTPUT_LABELS);

		MultiLayerNetwork network = new MultiLayerNetwork(configuration);
		network.setListeners(new PerformanceListener(1));
		network.init();

		while (true) {
			// features shape: [2, 1, 16, 96, 96] [minibatchsize, channels, depth, height, width]
			// labels shape: [2, 2] [minibatchsize, vector array size]
			DataSet dataSet = randomProvider.generateNextDataSet();

			network.fit(dataSet);
		}

	}

}
