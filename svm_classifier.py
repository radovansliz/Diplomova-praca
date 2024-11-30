import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv("12k_samples_12_families.csv")
print("Dataset Loaded Successfully")
print("First 5 Rows of the Dataset:")
print(data.head())

# Step 2: Show information about the dataset
print("\nDataset Information:")
print(data.info())

dynamic_columns = [
    "Memory_PssTotal", "Memory_PssClean", "Memory_SharedDirty", "Memory_PrivateDirty", "Memory_SharedClean",
    "Memory_PrivateClean", "Memory_SwapPssDirty", "Memory_HeapSize", "Memory_HeapAlloc", "Memory_HeapFree",
    "Memory_Views", "Memory_ViewRootImpl", "Memory_AppContexts", "Memory_Activities", "Memory_Assets",
    "Memory_AssetManagers", "Memory_LocalBinders", "Memory_ProxyBinders", "Memory_ParcelMemory",
    "Memory_ParcelCount", "Memory_DeathRecipients", "Memory_OpenSSLSockets", "Memory_WebViews",
    "API_Process_android.os.Process_start", "API_Process_android.app.ActivityManager_killBackgroundProcesses",
    "API_Process_android.os.Process_killProcess", "API_Command_java.lang.Runtime_exec",
    "API_Command_java.lang.ProcessBuilder_start", "API_JavaNativeInterface_java.lang.Runtime_loadLibrary",
    "API_JavaNativeInterface_java.lang.Runtime_load", "API_WebView_android.webkit.WebView_loadUrl",
    "API_WebView_android.webkit.WebView_loadData", "API_WebView_android.webkit.WebView_loadDataWithBaseURL",
    "API_WebView_android.webkit.WebView_addJavascriptInterface", "API_WebView_android.webkit.WebView_evaluateJavascript",
    "API_WebView_android.webkit.WebView_postUrl", "API_WebView_android.webkit.WebView_postWebMessage",
    "API_WebView_android.webkit.WebView_savePassword", "API_WebView_android.webkit.WebView_setHttpAuthUsernamePassword",
    "API_WebView_android.webkit.WebView_getHttpAuthUsernamePassword", "API_WebView_android.webkit.WebView_setWebContentsDebuggingEnabled",
    "API_FileIO_libcore.io.IoBridge_open", "API_FileIO_android.content.ContextWrapper_openFileInput",
    "API_FileIO_android.content.ContextWrapper_openFileOutput", "API_FileIO_android.content.ContextWrapper_deleteFile",
    "API_Database_android.content.ContextWrapper_openOrCreateDatabase", "API_Database_android.content.ContextWrapper_databaseList",
    "API_Database_android.content.ContextWrapper_deleteDatabase", "API_Database_android.database.sqlite.SQLiteDatabase_execSQL",
    "API_Database_android.database.sqlite.SQLiteDatabase_deleteDatabase", "API_Database_android.database.sqlite.SQLiteDatabase_getPath",
    "API_Database_android.database.sqlite.SQLiteDatabase_insert", "API_Database_android.database.sqlite.SQLiteDatabase_insertOrThrow",
    "API_Database_android.database.sqlite.SQLiteDatabase_insertWithOnConflict", "API_Database_android.database.sqlite.SQLiteDatabase_openDatabase",
    "API_Database_android.database.sqlite.SQLiteDatabase_openOrCreateDatabase", "API_Database_android.database.sqlite.SQLiteDatabase_query",
    "API_Database_android.database.sqlite.SQLiteDatabase_queryWithFactory", "API_Database_android.database.sqlite.SQLiteDatabase_rawQuery",
    "API_Database_android.database.sqlite.SQLiteDatabase_rawQueryWithFactory", "API_Database_android.database.sqlite.SQLiteDatabase_update",
    "API_Database_android.database.sqlite.SQLiteDatabase_updateWithOnConflict", "API_Database_android.database.sqlite.SQLiteDatabase_compileStatement",
    "API_Database_android.database.sqlite.SQLiteDatabase_create", "API_IPC_android.content.ContextWrapper_sendBroadcast",
    "API_IPC_android.content.ContextWrapper_sendStickyBroadcast", "API_IPC_android.content.ContextWrapper_startActivity",
    "API_IPC_android.content.ContextWrapper_startService", "API_IPC_android.content.ContextWrapper_stopService",
    "API_IPC_android.content.ContextWrapper_registerReceiver", "API_Binder_android.app.ContextImpl_registerReceiver",
    "API_Binder_android.app.ActivityThread_handleReceiver", "API_Binder_android.app.Activity_startActivity",
    "API_Crypto_javax.crypto.spec.SecretKeySpec_$init", "API_Crypto_javax.crypto.Cipher_doFinal",
    "API_Crypto-Hash_java.security.MessageDigest_digest", "API_Crypto-Hash_java.security.MessageDigest_update",
    "API_DeviceInfo_android.telephony.TelephonyManager_getDeviceId", "API_DeviceInfo_android.telephony.TelephonyManager_getSubscriberId",
    "API_DeviceInfo_android.telephony.TelephonyManager_getLine1Number", "API_DeviceInfo_android.telephony.TelephonyManager_getNetworkOperator",
    "API_DeviceInfo_android.telephony.TelephonyManager_getNetworkOperatorName", "API_DeviceInfo_android.telephony.TelephonyManager_getSimOperatorName",
    "API_DeviceInfo_android.net.wifi.WifiInfo_getMacAddress", "API_DeviceInfo_android.net.wifi.WifiInfo_getBSSID",
    "API_DeviceInfo_android.net.wifi.WifiInfo_getIpAddress", "API_DeviceInfo_android.net.wifi.WifiInfo_getNetworkId",
    "API_DeviceInfo_android.telephony.TelephonyManager_getSimCountryIso", "API_DeviceInfo_android.telephony.TelephonyManager_getSimSerialNumber",
    "API_DeviceInfo_android.telephony.TelephonyManager_getNetworkCountryIso", "API_DeviceInfo_android.telephony.TelephonyManager_getDeviceSoftwareVersion",
    "API_DeviceInfo_android.os.Debug_isDebuggerConnected", "API_DeviceInfo_android.content.pm.PackageManager_getInstallerPackageName",
    "API_DeviceInfo_android.content.pm.PackageManager_getInstalledApplications", "API_DeviceInfo_android.content.pm.PackageManager_getInstalledModules",
    "API_DeviceInfo_android.content.pm.PackageManager_getInstalledPackages", "API_Network_java.net.URL_openConnection",
    "API_Network_org.apache.http.impl.client.AbstractHttpClient_execute", "API_Network_com.android.okhttp.internal.huc.HttpURLConnectionImpl_getInputStream",
    "API_Network_com.android.okhttp.internal.http.HttpURLConnectionImpl_getInputStream", "API_DexClassLoader_dalvik.system.BaseDexClassLoader_findResource",
    "API_DexClassLoader_dalvik.system.BaseDexClassLoader_findResources", "API_DexClassLoader_dalvik.system.BaseDexClassLoader_findLibrary",
    "API_DexClassLoader_dalvik.system.DexFile_loadDex", "API_DexClassLoader_dalvik.system.DexFile_loadClass",
    "API_DexClassLoader_dalvik.system.DexClassLoader_$init", "API_Base64_android.util.Base64_decode", "API_Base64_android.util.Base64_encode",
    "API_Base64_android.util.Base64_encodeToString", "API_SystemManager_android.app.ApplicationPackageManager_setComponentEnabledSetting",
    "API_SystemManager_android.app.NotificationManager_notify", "API_SystemManager_android.telephony.TelephonyManager_listen",
    "API_SystemManager_android.content.BroadcastReceiver_abortBroadcast", "API_SMS_android.telephony.SmsManager_sendTextMessage",
    "API_SMS_android.telephony.SmsManager_sendMultipartTextMessage", "API_DeviceData_android.content.ContentResolver_query",
    "API_DeviceData_android.content.ContentResolver_registerContentObserver", "API_DeviceData_android.content.ContentResolver_insert",
    "API_DeviceData_android.content.ContentResolver_delete", "API_DeviceData_android.accounts.AccountManager_getAccountsByType",
    "API_DeviceData_android.accounts.AccountManager_getAccounts", "API_DeviceData_android.location.Location_getLatitude",
    "API_DeviceData_android.location.Location_getLongitude", "API_DeviceData_android.media.AudioRecord_startRecording",
    "API_DeviceData_android.media.MediaRecorder_start", "API_DeviceData_android.os.SystemProperties_get",
    "API_DeviceData_android.app.ApplicationPackageManager_getInstalledPackages", "API__sessions", "Network_TotalReceivedBytes",
    "Network_TotalReceivedPackets", "Network_TotalTransmittedBytes", "Network_TotalTransmittedPackets", "Battery_wakelock",
    "Battery_service", "Logcat_verbose", "Logcat_debug", "Logcat_info", "Logcat_warning", "Logcat_error", "Logcat_total",
    "Process_total"
]

# Step 3: Extract features and labels
X = data[dynamic_columns]
# X = data.drop(columns=["Hash", "Category", "Family"])
y = data["Family"]

# Show the extracted features and labels
print("\nFeatures (X) Preview:")
print(X.head())
print("\nLabels (y) Preview:")
print(y.head())

# Step 4: Vectorize the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("\nLabels After Encoding:")
print(y[:5])

# Step 5: Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("\nFeatures After Standardization (First 5 Rows):")
print(X[:5])

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nData Split into Training and Testing Sets")
print(f"Training Set Size: {X_train.shape[0]} samples")
print(f"Testing Set Size: {X_test.shape[0]} samples")

# Step 7: Initialize the SVM model with SGD (Support Vector Machine with hinge loss)
svm_classifier = SGDClassifier(loss="hinge", max_iter=1, tol=None, warm_start=True)
train_losses = []
val_losses = []

# Step 8: Train the model with loss tracking
for i in range(1, 301):  # 300 epochs for better loss tracking
    svm_classifier.max_iter = i  # Update the number of iterations
    svm_classifier.fit(X_train, y_train)
    
    # Calculate training loss
    train_loss = 1 - svm_classifier.score(X_train, y_train)  # Using hinge loss equivalent
    train_losses.append(train_loss)
    
    # Calculate validation loss
    val_loss = 1 - svm_classifier.score(X_test, y_test)  # Using hinge loss equivalent
    val_losses.append(val_loss)

# Step 9: Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 301), train_losses, label='Training Loss')
plt.plot(range(1, 301), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('1 - Accuracy (Proxy for Loss)')
plt.title('Training and Validation Loss Curve for SVM')
plt.legend()
plt.savefig("svm_loss_curve.png")
plt.close()

# Step 10: Evaluate the SVM model
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 11: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 12: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot and save the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("svm_confusion_matrix.png")  # Save the confusion matrix plot
plt.close()  # Close the plot to prevent displaying it
