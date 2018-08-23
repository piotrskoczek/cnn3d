package com.pskoczek.cnn3d;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.NDArrayUtil;

public class RandomDataSetProvider {
    public int height = 96, width = 96, channels = 1, depth = 16, outputLabels;
    private int minibatches = 10;
   
    private Random rnd = new Random();
    
    public RandomDataSetProvider(int height, int width, int channels, int depth, int minibatches, int outputLabels) {
    	this.height=height;
    	this.width=width;
    	this.channels=channels;
    	this.depth=depth;
    	this.minibatches=minibatches;
    	this.outputLabels=outputLabels;
    }

    public DataSet generateNextDataSet()
    {
        INDArray miniBatchFeatures = Nd4j.create( minibatches, 16, 96, 96, 1 );
        
        for (int i = 0; i < minibatches; i++) {
            INDArray feature = generateSingleFeature();
            miniBatchFeatures.putRow( i, feature );
        }

        INDArray labels = Nd4j.create( minibatches, 2 );
        
        for (int i = 0; i < minibatches; i++) {
            int index = rnd.nextInt( 2 );
            int[] labelsSimpleArray = new int[ 2 ];
            labelsSimpleArray[ index ] = 1;

            labels.putRow( i, NDArrayUtil.toNDArray( labelsSimpleArray ) );
        }

        INDArray reshapedFeatures = miniBatchFeatures.permute( 0, 4, 1, 2, 3 );
        
        return new DataSet( reshapedFeatures, labels );
    }
    

    private INDArray generateSingleFeature()
    {
        INDArray timeArray = Nd4j.create( 16, 96, 96, 1 );
        for (int i = 0; i < 16; i++) {
            double[][] generatedPicture = generateRandomPicture();
            INDArray pictureArray = Nd4j.create( generatedPicture );
            pictureArray = pictureArray.reshape( 96, 96, 1 );
            timeArray.putRow( i, pictureArray );
        }

        return timeArray;
    }

    private double[][] generateRandomPicture()
    {
        double[][] result = new double[ height ][ width ];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[ i ].length; j++) {
                result[ i ][ j ] = (double) rnd.nextInt( 255 ) / (double) 255;
            }
        }

        return result;
    }

}
