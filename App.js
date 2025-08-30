import React, { useState, useEffect, useCallback, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, SafeAreaView, StatusBar, Alert } from 'react-native';
import { Accelerometer } from 'expo-sensors';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

import modelJSON from './assets/final_model.json';

// --- Constants ---
const MINIMUM_RECORDING_DURATION = 10; // in seconds
const WINDOW_SIZE = 51; // 51 samples ~ 2.56s at 20Hz (50ms interval)
const STEP_SIZE = 25;   // ~50% overlap, creates a new prediction roughly every 1.25s
const ACCELEROMETER_INTERVAL_MS = 50; // 20 Hz, matching the recording rate

// --- On-Device Prediction Engine ---
// This engine reads the JSON model and makes live predictions
const RandomForestEngine = {
  predict: (features) => {
    // Basic checks to ensure the model and features are valid
    if (!modelJSON || !modelJSON.trees || features.length !== modelJSON.featureOrder.length) {
      console.error("Model not loaded or feature length mismatch!");
      return "Error"; // Return an error state
    }

    // Get a prediction from each tree in the forest
    const treePredictions = modelJSON.trees.map(tree => {
      let currentNodeIndex = 0;
      const nodes = tree.nodes;
      
      // Traverse the tree from the root to a leaf node
      while (nodes[currentNodeIndex].leftChild !== -1) {
        const node = nodes[currentNodeIndex];
        const featureValue = features[node.featureIndex];
        
        if (featureValue <= node.threshold) {
          currentNodeIndex = node.leftChild;
        } else {
          currentNodeIndex = node.rightChild;
        }
      }
      // The leaf node contains the prediction
      return nodes[currentNodeIndex].predictedClassIndex;
    });

    // Perform a "majority vote" to get the final prediction
    const votes = treePredictions.reduce((acc, val) => {
      acc[val] = (acc[val] || 0) + 1;
      return acc;
    }, {});
    
    const majorityVoteIndex = Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);
    return modelJSON.classes[majorityVoteIndex];
  }
};

// --- Feature Engineering ---
function extractFeaturesFromWindow(window) {
    const features = [];
    const axes = ['x', 'y', 'z'];
    
    // Helper functions for math
    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = (arr, avg) => Math.sqrt(arr.map(x => Math.pow(x - avg, 2)).reduce((a, b) => a + b) / arr.length);

    // 1. Basic & Range Features
    axes.forEach(axis => {
        const signal = window.map(d => d[axis]);
        const m = mean(signal);
        const minVal = Math.min(...signal);
        const maxVal = Math.max(...signal);
        
        features.push(m);                       // mean
        features.push(std(signal, m));          // std
        features.push(minVal);                  // min
        features.push(maxVal);                  // max
        features.push(maxVal - minVal);         // range
    });

    // 2. Magnitude Features
    const magnitude = window.map(d => Math.sqrt(d.x**2 + d.y**2 + d.z**2));
    const meanMag = mean(magnitude);
    features.push(meanMag);                     // mean_magnitude
    features.push(std(magnitude, meanMag));     // std_magnitude

    // 3. Jerk Features
    axes.forEach(axis => {
        const signal = window.map(d => d[axis]);
        const jerk = [];
        for (let i = 1; i < signal.length; i++) {
            jerk.push(signal[i] - signal[i - 1]);
        }
        // Handle case where jerk is empty if window is too small
        const jerkMean = jerk.length > 0 ? mean(jerk) : 0;
        const jerkStd = jerk.length > 0 ? std(jerk, jerkMean) : 0;
        features.push(jerkStd);  // std_jerk
    });
    
    // The order of feature pushing MUST match the `featureOrder` array in the JSON model
    return features;
}

// --- Main App Component ---
export default function App() {
  const [mode, setMode] = useState('tracker'); // 'tracker' or 'collector'

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.nav}>
        <TouchableOpacity
          style={[styles.navButton, mode === 'tracker' && styles.navButtonActive]}
          onPress={() => setMode('tracker')}
        >
          <Text style={styles.navButtonText}>Tracker Mode</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.navButton, mode === 'collector' && styles.navButtonActive]}
          onPress={() => setMode('collector')}
        >
          <Text style={styles.navButtonText}>Collector Mode</Text>
        </TouchableOpacity>
      </View>
      {mode === 'tracker' ? <TrackerScreen /> : <CollectorScreen />}
    </SafeAreaView>
  );
}

// --- Mode 1: Tracker Screen ---
const TrackerScreen = () => {
  const dataBuffer = useRef([]);
  const [currentClass, setCurrentClass] = useState('Sedentary');
  const [time, setTime] = useState({ Sedentary: 0, Light: 0, Moderate: 0, Vigorous: 0 });

  useEffect(() => {
    const subscription = Accelerometer.addListener(accelerometerData => {
        dataBuffer.current.push(accelerometerData);
        
        // When the buffer has enough data to form a complete window...
        if (dataBuffer.current.length >= WINDOW_SIZE) {
            const windowToProcess = dataBuffer.current.slice(0, WINDOW_SIZE);
            
            // 1. Run the on-device feature engineering
            const features = extractFeaturesFromWindow(windowToProcess);
            
            // 2. Get a prediction from the live model
            const predictedClass = RandomForestEngine.predict(features);
            setCurrentClass(predictedClass);

            // 3. Slide the buffer forward to prepare for the next prediction
            dataBuffer.current.splice(0, STEP_SIZE);
        }
    });

    // Set the sensor to our desired 20Hz rate
    Accelerometer.setUpdateInterval(ACCELEROMETER_INTERVAL_MS);
    
    return () => subscription.remove();
  }, []);

  // Timer for updating cumulative time
  useEffect(() => {
    const timer = setInterval(() => {
      setTime(prevTime => ({
        ...prevTime,
        [currentClass]: prevTime[currentClass] + 1,
      }));
    }, 1000);
    return () => clearInterval(timer);
  }, [currentClass]);

  const formatTime = (seconds) => {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
  };

  const MET_CLASSES = [
    { name: 'Sedentary', color: '#4A90E2', desc: '< 1.5 METs' },
    { name: 'Light', color: '#50E3C2', desc: '1.5 - 3 METs' },
    { name: 'Moderate', color: '#F5A623', desc: '3 - 6 METs' },
    { name: 'Vigorous', color: '#D0021B', desc: '> 6 METs' },
  ];

  return (
    <View style={styles.screenContainer}>
      <Text style={styles.header}>Today's Activity</Text>
      <View style={styles.cardContainer}>
        {MET_CLASSES.map(metClass => (
          <View key={metClass.name} style={[styles.card, { borderColor: metClass.color }]}>
            <Text style={[styles.cardTitle, { color: metClass.color }]}>{metClass.name}</Text>
            <Text style={styles.cardDesc}>{metClass.desc}</Text>
            <Text style={styles.cardTime}>{formatTime(time[metClass.name])}</Text>
          </View>
        ))}
      </View>
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>Current Activity:</Text>
        <Text style={[styles.statusValue, { color: MET_CLASSES.find(c => c.name === currentClass)?.color || '#fff' }]}>
          {currentClass}
        </Text>
      </View>
    </View>
  );
};


// --- Mode 2: Data Collector Screen ---
const RECORDING_DIR = FileSystem.documentDirectory + 'recordings/';
let recordingData = [];

const CollectorScreen = () => {
  const [selectedActivity, setSelectedActivity] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [savedFiles, setSavedFiles] = useState([]);
  const [showWarning, setShowWarning] = useState(false);
  const movementBuffer = useRef([]);
  const noMovementCounter = useRef(0);

  const ensureDirExists = async () => {
    const dirInfo = await FileSystem.getInfoAsync(RECORDING_DIR);
    if (!dirInfo.exists) {
      await FileSystem.makeDirectoryAsync(RECORDING_DIR, { intermediates: true });
    }
  };

  const refreshFileList = useCallback(async () => {
    await ensureDirExists();
    const files = await FileSystem.readDirectoryAsync(RECORDING_DIR);
    setSavedFiles(files.sort().reverse());
  }, []);

  useEffect(() => {
    refreshFileList();
  }, [refreshFileList]);

  useEffect(() => {
    let timer;
    if (isRecording) {
      timer = setInterval(() => setDuration(d => d + 1), 1000);
    }
    return () => clearInterval(timer);
  }, [isRecording]);

  useEffect(() => {
    let subscription;
    if (isRecording) {
      subscription = Accelerometer.addListener(data => {
        recordingData.push({ ...data, timestamp: Date.now() });
        if (selectedActivity === 'Moderate' || selectedActivity === 'Vigorous') {
          movementBuffer.current.push(Math.sqrt(data.x**2 + data.y**2 + data.z**2));
          if (movementBuffer.current.length > 20) {
            movementBuffer.current.shift();
            const stdDev = getStandardDeviation(movementBuffer.current);
            if (stdDev < 0.05) {
              noMovementCounter.current += 1;
            } else {
              noMovementCounter.current = 0;
            }
            if (noMovementCounter.current > 60) {
              setShowWarning(true);
            } else {
              setShowWarning(false);
            }
          }
        }
      });
      Accelerometer.setUpdateInterval(ACCELEROMETER_INTERVAL_MS);
    }
    return () => subscription?.remove();
  }, [isRecording, selectedActivity]);

  const getStandardDeviation = (array) => {
    const n = array.length;
    if (n === 0) return 0;
    const mean = array.reduce((a, b) => a + b) / n;
    return Math.sqrt(array.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n);
  };
  
  const startRecording = () => {
    if (!selectedActivity) {
      Alert.alert('Select Activity', 'Please select an activity to record first.');
      return;
    }
    recordingData = [];
    movementBuffer.current = [];
    noMovementCounter.current = 0;
    setShowWarning(false);
    setDuration(0);
    setIsRecording(true);
  };

  const stopAndSaveRecording = async () => {
    setIsRecording(false);
    setShowWarning(false);
    if (duration < MINIMUM_RECORDING_DURATION) {
        Alert.alert('Recording Too Short', `Please record for at least ${MINIMUM_RECORDING_DURATION} seconds.`);
        recordingData = [];
        return;
    }
    if (recordingData.length === 0) return;
    const sessionId = `${Date.now()}_${selectedActivity}`;
    const filename = `${RECORDING_DIR}${sessionId}.csv`;
    const csvHeader = 'timestamp,x,y,z,label,session_id\n';
    const csvRows = recordingData.map(d =>
      `${d.timestamp},${d.x},${d.y},${d.z},${selectedActivity},${sessionId}`
    ).join('\n');
    await FileSystem.writeAsStringAsync(filename, csvHeader + csvRows);
    recordingData = [];
    refreshFileList();
  };
  
  const handleShare = async (filename) => {
      const fileUri = RECORDING_DIR + filename;
      if (!(await Sharing.isAvailableAsync())) {
          Alert.alert('Sharing Not Available', 'Sharing is not available on this device.');
          return;
      }
      try {
          await Sharing.shareAsync(fileUri);
      } catch (error) {
          Alert.alert('Error', 'Failed to share the file.');
      }
  };

  const handleDelete = (filename) => {
      Alert.alert(
          "Delete Recording",
          `Are you sure you want to permanently delete ${filename}?`,
          [
              { text: "Cancel", style: "cancel" },
              { 
                  text: "Delete", 
                  style: "destructive", 
                  onPress: async () => {
                      try {
                          await FileSystem.deleteAsync(RECORDING_DIR + filename);
                          refreshFileList();
                      } catch (error) {
                          Alert.alert("Error", "Failed to delete the file.");
                      }
                  }
              }
          ]
      );
  };

  const handleShareAll = async () => {
    if (savedFiles.length === 0) {
      Alert.alert('No Recordings', 'There are no recordings to export.');
      return;
    }
    try {
      const fileUris = savedFiles.map(file => RECORDING_DIR + file);
      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(fileUris, { dialogTitle: 'Export All Recordings' });
      } else {
        Alert.alert('Sharing Not Available');
      }
    } catch (error) {
      console.error("Error sharing files:", error);
      Alert.alert("Export Failed", "Could not share the recordings.");
    }
  };

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };
  
  const ACTIVITY_TYPES = ['Sedentary', 'Light', 'Moderate', 'Vigorous'];

  return (
    <View style={styles.screenContainer}>
        {isRecording && (
          <View style={styles.recordingBanner}>
            <Text style={styles.recordingBannerText}>RECORDING: {selectedActivity}</Text>
          </View>
        )}
        <Text style={styles.header}>Data Collector</Text>
        <View style={styles.collectorControls}>
            <Text style={styles.collectorLabel}>1. Select Activity to Record</Text>
            <View style={styles.activityButtons}>
                {ACTIVITY_TYPES.map(activity => (
                    <TouchableOpacity 
                        key={activity}
                        style={[
                            styles.activityButton, 
                            selectedActivity === activity && styles.activityButtonSelected,
                            isRecording && styles.disabledButton
                        ]}
                        onPress={() => setSelectedActivity(activity)}
                        disabled={isRecording}
                    >
                        <Text style={styles.activityButtonText}>{activity}</Text>
                    </TouchableOpacity>
                ))}
            </View>
            <Text style={styles.collectorLabel}>2. Start & Stop Recording</Text>
            <TouchableOpacity 
                style={[styles.recordButton, isRecording ? styles.recordButtonStop : styles.recordButtonStart]}
                onPress={isRecording ? stopAndSaveRecording : startRecording}
            >
                <Text style={styles.recordButtonText}>{isRecording ? `STOP (${formatTime(duration)})` : 'START RECORDING'}</Text>
            </TouchableOpacity>
            {showWarning && <Text style={styles.warningText}>Warning: No movement detected!</Text>}
        </View>

        <View style={styles.fileListContainer}>
            <View style={styles.fileListHeader}>
                <Text style={styles.header}>Saved Recordings</Text>
                <TouchableOpacity style={styles.exportButton} onPress={handleShareAll}>
                    <Text style={styles.exportButtonText}>Export All</Text>
                </TouchableOpacity>
            </View>
            {savedFiles.length === 0 ? (
                <Text style={styles.noFilesText}>Record sessions of 10s or more to see files here.</Text>
            ) : (
                <ScrollView>
                    {savedFiles.map(file => (
                        <View key={file} style={styles.fileItem}>
                            <Text style={styles.fileName} numberOfLines={1}>{file}</Text>
                            <View style={styles.fileActions}>
                                <TouchableOpacity style={styles.fileButton} onPress={() => handleShare(file)}>
                                    <Text style={styles.fileButtonText}>Share</Text>
                                </TouchableOpacity>
                                <TouchableOpacity style={[styles.fileButton, styles.deleteButton]} onPress={() => handleDelete(file)}>
                                    <Text style={styles.fileButtonText}>Delete</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    ))}
                </ScrollView>
            )}
        </View>
    </View>
  );
};


// --- Styles ---
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212' },
  screenContainer: { flex: 1, padding: 20 },
  header: { fontSize: 26, fontWeight: 'bold', color: '#FFFFFF', marginBottom: 20, textAlign: 'center' },
  nav: { flexDirection: 'row', backgroundColor: '#1E1E1E' },
  navButton: { flex: 1, padding: 15, alignItems: 'center', justifyContent: 'center', borderBottomWidth: 3, borderBottomColor: 'transparent' },
  navButtonActive: { borderBottomColor: '#BB86FC' },
  navButtonText: { color: '#FFFFFF', fontSize: 16, fontWeight: '600' },
  cardContainer: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
  card: { width: '48%', backgroundColor: '#1E1E1E', padding: 20, borderRadius: 12, marginBottom: 15, borderWidth: 2 },
  cardTitle: { fontSize: 18, fontWeight: 'bold' },
  cardDesc: { fontSize: 14, color: '#A0A0A0', marginTop: 4 },
  cardTime: { fontSize: 24, color: '#FFFFFF', fontWeight: '700', marginTop: 12 },
  statusContainer: { marginTop: 20, padding: 20, backgroundColor: '#1E1E1E', borderRadius: 12, alignItems: 'center' },
  statusText: { fontSize: 18, color: '#A0A0A0' },
  statusValue: { fontSize: 28, fontWeight: 'bold', marginTop: 8 },
  recordingBanner: { position: 'absolute', top: 0, left: 0, right: 0, backgroundColor: '#BB86FC', padding: 10, alignItems: 'center', zIndex: 10 },
  recordingBannerText: { color: '#121212', fontSize: 16, fontWeight: 'bold' },
  collectorControls: { backgroundColor: '#1E1E1E', padding: 20, borderRadius: 12, marginBottom: 20 },
  collectorLabel: { fontSize: 16, color: '#A0A0A0', marginBottom: 10 },
  activityButtons: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', marginBottom: 20 },
  activityButton: { backgroundColor: '#333333', paddingVertical: 10, paddingHorizontal: 15, borderRadius: 20, marginBottom: 10, minWidth: '48%', alignItems: 'center' },
  activityButtonSelected: { backgroundColor: '#BB86FC' },
  activityButtonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '600' },
  recordButton: { padding: 20, borderRadius: 12, alignItems: 'center' },
  recordButtonStart: { backgroundColor: '#03DAC6' },
  recordButtonStop: { backgroundColor: '#CF6679' },
  recordButtonText: { color: '#121212', fontSize: 18, fontWeight: 'bold' },
  disabledButton: { opacity: 0.5 },
  warningText: { color: '#CF6679', textAlign: 'center', marginTop: 10, fontWeight: 'bold' },
  fileListContainer: { flex: 1, backgroundColor: '#1E1E1E', borderRadius: 12, padding: 15 },
  fileListHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  exportButton: { backgroundColor: '#333', paddingVertical: 8, paddingHorizontal: 14, borderRadius: 8 },
  exportButtonText: { color: '#FFFFFF', fontWeight: '600' },
  noFilesText: { color: '#A0A0A0', textAlign: 'center', marginTop: 20 },
  fileItem: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#333' },
  fileName: { color: '#FFFFFF', fontSize: 14, flex: 1, marginRight: 10 },
  fileActions: { flexDirection: 'row' },
  fileButton: { marginLeft: 10, backgroundColor: '#333', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 6 },
  deleteButton: { backgroundColor: '#CF6679' },
  fileButtonText: { color: '#FFFFFF', fontSize: 12 },
});
