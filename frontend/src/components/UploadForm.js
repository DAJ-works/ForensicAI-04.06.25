import React, { useState, useEffect } from 'react';
import { 
  Typography, Box, Button, Paper, LinearProgress, Alert, 
  TextField, Grid, Card, CardContent, IconButton,
  Stepper, Step, StepLabel,
  Divider, useTheme, Tooltip, Backdrop, CircularProgress
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';

// Icons
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import VideoFileIcon from '@mui/icons-material/VideoFile';
import EditIcon from '@mui/icons-material/Edit';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

// Styled components
const UploadContainer = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  maxWidth: 900,
  margin: '40px auto',
  borderRadius: 16,
  boxShadow: '0 8px 40px rgba(0, 0, 0, 0.12)',
  overflow: 'hidden',
  position: 'relative',
  background: theme.palette.mode === 'dark' 
    ? `linear-gradient(145deg, ${theme.palette.grey[900]}, ${theme.palette.grey[800]})`
    : `linear-gradient(145deg, #ffffff, #f5f7fa)`
}));

const StyledStepper = styled(Stepper)(({ theme }) => ({
  '& .MuiStepConnector-line': {
    minHeight: 12,
  },
  '& .MuiStepIcon-root': {
    color: theme.palette.grey[400],
    '&.Mui-active': {
      color: theme.palette.primary.main,
    },
    '&.Mui-completed': {
      color: theme.palette.success.main,
    }
  }
}));

const DropZone = styled(Box)(({ theme, isDragActive, hasFile }) => ({
  border: `2px dashed ${isDragActive ? theme.palette.primary.main : hasFile ? theme.palette.success.main : theme.palette.grey[300]}`,
  borderRadius: 12,
  padding: theme.spacing(5, 2),
  backgroundColor: isDragActive ? 
    alpha(theme.palette.primary.main, 0.05) : 
    hasFile ? alpha(theme.palette.success.main, 0.05) : 'transparent',
  transition: 'all 0.3s ease',
  outline: 'none',
  cursor: 'pointer',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: 200
}));

const VideoPreviewCard = styled(Card)(({ theme }) => ({
  margin: theme.spacing(2, 0),
  borderRadius: 8,
  overflow: 'hidden',
  backgroundColor: alpha(theme.palette.background.paper, 0.6),
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: theme.shadows[6]
  }
}));

const StyledProgress = styled(LinearProgress)(({ theme, value }) => ({
  height: 8,
  borderRadius: 4,
  backgroundColor: alpha(theme.palette.primary.main, 0.15),
  '& .MuiLinearProgress-bar': {
    borderRadius: 4,
    background: value === 100 
      ? theme.palette.success.main 
      : `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
  }
}));

const SuccessButton = styled(Button)(({ theme }) => ({
  background: `linear-gradient(45deg, ${theme.palette.success.main} 30%, ${theme.palette.success.light} 90%)`,
  boxShadow: '0 3px 10px 2px rgba(76, 175, 80, 0.3)',
  color: theme.palette.common.white,
  '&:hover': {
    boxShadow: '0 5px 14px 4px rgba(76, 175, 80, 0.4)',
  }
}));

// Hidden input for file selection
const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

// Helper functions
const formatBytes = (bytes, decimals = 2) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

// Helper for MUI alpha
const alpha = (color, value) => {
  return color ? color.replace('rgb', 'rgba').replace(')', `, ${value})`) : color;
};

// Main component
const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [caseName, setCaseName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [caseId, setCaseId] = useState(null);
  const [eventSource, setEventSource] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  
  const navigate = useNavigate();
  const theme = useTheme();

  // Clean up event source on unmount
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
      }
      // Clean up any video preview URLs
      if (videoPreview) {
        URL.revokeObjectURL(videoPreview);
      }
    };
  }, [eventSource, videoPreview]);

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  // Process the selected file
  const processFile = (selectedFile) => {
    // Generate preview for video files
    if (selectedFile.type.includes('video')) {
      const preview = URL.createObjectURL(selectedFile);
      setVideoPreview(preview);
    }
    
    setFile(selectedFile);
    
    // Generate a default case name from the file name
    const fileName = selectedFile.name.split('.')[0]
      .replace(/_/g, ' ')
      .replace(/-/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase()); // Capitalize first letter of each word
    
    setCaseName(fileName);
    setActiveStep(1); // Move to step 2
    setError('');
  };

  // Simple drag and drop implementation
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // Handle case name change
  const handleCaseNameChange = (event) => {
    setCaseName(event.target.value);
  };

  // Set up SSE listener for analysis completion
  const listenForAnalysisCompletion = (caseId) => {
    try {
      const sse = new EventSource(`/api/events/analysis/${caseId}`);
      
      sse.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.status === 'processing') {
            setAnalysisProgress(Math.max(5, data.progress || 0)); // Always show minimum 5% progress for feedback
          } else if (data.status === 'completed') {
            setAnalysisProgress(100);
            setAnalyzing(false);
            setSuccess(true);
            setActiveStep(3);
            sse.close();
            
            // Navigate to the case detail page after a short delay
            setTimeout(() => {
              navigate(`/cases/${caseId}`);
            }, 2000);
          } else if (data.status === 'failed') {
            setError(`Analysis failed: ${data.error || 'Unknown error'}`);
            setAnalyzing(false);
            sse.close();
          }
        } catch (e) {
          console.error("Error parsing SSE message:", e);
        }
      };
      
      sse.onerror = () => {
        console.error("SSE connection error");
        // Don't close the connection on first error, let the browser retry
      };
      
      setEventSource(sse);
    } catch (error) {
      console.error("Error setting up SSE connection:", error);
    }
  };

  // Clear selected file
  const handleClearFile = () => {
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
    setFile(null);
    setVideoPreview(null);
    setActiveStep(0);
  };

  // Upload and analyze video
  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file');
      return;
    }

    if (!caseName.trim()) {
      setError('Please enter a case name');
      return;
    }

    setError('');
    setUploading(true);
    setUploadProgress(0);
    setIsLoading(true);

    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('case_name', caseName);

    try {
      // Upload file with simulated progress until we get response
      const uploadSimulation = setInterval(() => {
        setUploadProgress(prev => {
          if (prev < 90) return prev + (Math.random() * 5);
          return prev;
        });
      }, 300);

      const uploadResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(uploadSimulation);
      setUploadProgress(100);

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const uploadData = await uploadResponse.json();
      setCaseId(uploadData.case_id);
      
      // Start analysis
      setActiveStep(2);
      setAnalyzing(true);
      setAnalysisProgress(0);

      const analyzeResponse = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          case_id: uploadData.case_id,
          case_name: caseName,
          video_path: uploadData.video_path
        }),
      });

      if (!analyzeResponse.ok) {
        const errorData = await analyzeResponse.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const analyzeData = await analyzeResponse.json();
      setIsLoading(false);
      
      // Start listening for analysis completion
      listenForAnalysisCompletion(analyzeData.case_id);
      
      // Redirect to cases page while analysis continues in background
      navigate('/cases', { state: { analysisPending: true, caseId: analyzeData.case_id } });
      
    } catch (error) {
      setError(`Error: ${error.message}`);
      setUploading(false);
      setAnalyzing(false);
      setIsLoading(false);
    }
  };

  // Animation classes using CSS transition
  const fadeIn = {
    opacity: 1,
    transition: 'opacity 0.5s ease, transform 0.5s ease',
    transform: 'translateY(0)'
  };

  const itemInitial = {
    opacity: 0,
    transform: 'translateY(20px)',
    transition: 'opacity 0.5s ease, transform 0.5s ease'
  };

  return (
    <Box sx={{ ...itemInitial, ...fadeIn }}>
      <UploadContainer elevation={4}>
        {/* Header */}
        <Box sx={{ mb: 4, textAlign: 'center', position: 'relative' }}>
          <Box sx={{ ...itemInitial, ...fadeIn, transitionDelay: '0.1s' }}>
            <Typography 
              variant="h4" 
              sx={{ 
                fontWeight: 600, 
                background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 1
              }}
            >
              Upload New Video Case
            </Typography>
            <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 3 }}>
              Upload surveillance footage for AI-powered analysis
            </Typography>
            <Divider sx={{ width: '50%', margin: '0 auto', mb: 4 }} />
          </Box>
        </Box>

        {/* Stepper */}
        <Box sx={{ ...itemInitial, ...fadeIn, transitionDelay: '0.2s' }}>
          <StyledStepper activeStep={activeStep} alternativeLabel sx={{ mb: 4 }}>
            {['Select Video', 'Case Details', 'Processing'].map((label, index) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </StyledStepper>
        </Box>

        {/* Error alert */}
        {error && (
          <Box sx={{ ...itemInitial, ...fadeIn, transitionDelay: '0.1s' }}>
            <Alert 
              severity="error" 
              sx={{ 
                mb: 3,
                borderRadius: 2,
                boxShadow: theme.shadows[2]
              }}
              onClose={() => setError('')}
            >
              {error}
            </Alert>
          </Box>
        )}

        {/* Success alert */}
        {success && (
          <Box sx={{ 
            opacity: 0, 
            transform: 'scale(0.95)', 
            animation: 'fadeInScale 0.5s forwards',
            '@keyframes fadeInScale': {
              to: { opacity: 1, transform: 'scale(1)' }
            }
          }}>
            <Alert 
              severity="success" 
              sx={{ 
                mb: 3,
                borderRadius: 2,
                boxShadow: theme.shadows[2]
              }}
            >
              <Typography variant="subtitle2">
                Video uploaded and analyzed successfully!
              </Typography>
            </Alert>
          </Box>
        )}

        {/* Main content - Step 1: File Selection */}
        {activeStep === 0 && (
          <Box sx={{ ...itemInitial, ...fadeIn, transitionDelay: '0.3s' }}>
            <DropZone 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              isDragActive={isDragOver} 
              hasFile={!!file}
              component="label"
            >
              <VisuallyHiddenInput 
                type="file" 
                accept="video/*" 
                onChange={handleFileChange}
                disabled={uploading || analyzing}
              />
              <Box sx={{ textAlign: 'center' }}>
                <CloudUploadIcon 
                  sx={{ 
                    fontSize: 64, 
                    color: isDragOver ? 'primary.main' : 'text.secondary',
                    mb: 2
                  }} 
                />
                <Typography variant="h6" gutterBottom>
                  {isDragOver 
                    ? 'Drop the video here...' 
                    : 'Drag & drop a video file here'}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Or click to browse your files
                </Typography>
                <Button 
                  variant="outlined" 
                  component="span"
                  startIcon={<VideoFileIcon />}
                  sx={{ borderRadius: 6 }}
                >
                  Select Video
                </Button>
                <Typography variant="caption" display="block" sx={{ mt: 2, color: 'text.secondary' }}>
                  Supported formats: MP4, AVI, MOV, etc.
                </Typography>
              </Box>
            </DropZone>
          </Box>
        )}

        {/* Step 2: File Selected & Case Details */}
        {activeStep === 1 && file && (
          <Box sx={{ ...itemInitial, ...fadeIn, transitionDelay: '0.3s' }}>
            <Grid container spacing={3}>
              {/* Video preview */}
              <Grid item xs={12} md={6}>
                <VideoPreviewCard>
                  <Box sx={{ position: 'relative', pb: 1 }}>
                    {videoPreview ? (
                      <Box sx={{ position: 'relative' }}>
                        <video 
                          src={videoPreview} 
                          style={{ 
                            width: '100%', 
                            maxHeight: '240px',
                            objectFit: 'cover',
                            borderRadius: '4px'
                          }} 
                          controls={false}
                        />
                        <Box sx={{ 
                          position: 'absolute', 
                          top: 0, 
                          left: 0, 
                          width: '100%', 
                          height: '100%', 
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          backgroundColor: 'rgba(0,0,0,0.3)',
                          borderRadius: '4px',
                        }}>
                          <IconButton 
                            sx={{ 
                              backgroundColor: 'rgba(255,255,255,0.9)',
                              '&:hover': {
                                backgroundColor: 'rgba(255,255,255,1)',
                                transform: 'scale(1.1)'
                              },
                              transition: 'transform 0.2s'
                            }}
                            onClick={() => window.open(videoPreview, '_blank')}
                          >
                            <PlayArrowIcon />
                          </IconButton>
                        </Box>
                      </Box>
                    ) : (
                      <Box sx={{ 
                        height: 200, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        backgroundColor: 'rgba(0,0,0,0.05)',
                        borderRadius: 1
                      }}>
                        <VideoFileIcon sx={{ fontSize: 64, color: 'text.secondary' }} />
                      </Box>
                    )}
                    <CardContent>
                      <Box sx={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center',
                        mb: 1
                      }}>
                        <Typography variant="subtitle1" component="div" sx={{ fontWeight: 500 }}>
                          {file.name.length > 30 ? file.name.substring(0, 30) + '...' : file.name}
                        </Typography>
                        <Tooltip title="Remove file">
                          <IconButton size="small" onClick={handleClearFile} color="error">
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {formatBytes(file.size)}
                      </Typography>
                    </CardContent>
                  </Box>
                </VideoPreviewCard>
              </Grid>

              {/* Case details */}
              <Grid item xs={12} md={6}>
                <Card sx={{ 
                  height: '100%', 
                  borderRadius: 2,
                  boxShadow: theme.shadows[2],
                  display: 'flex',
                  flexDirection: 'column'
                }}>
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                      <EditIcon sx={{ mr: 1, color: 'primary.main' }} fontSize="small" />
                      Case Details
                    </Typography>
                    <Divider sx={{ mb: 3 }} />
                    
                    <TextField
                      label="Case Name"
                      value={caseName}
                      onChange={handleCaseNameChange}
                      disabled={uploading || analyzing}
                      variant="outlined"
                      fullWidth
                      sx={{ mb: 2 }}
                      placeholder="Enter a descriptive name"
                      helperText="Give your case a descriptive name for easy reference"
                      InputProps={{
                        sx: { borderRadius: 2 }
                      }}
                    />
                    
                    <Box sx={{ mt: 3 }}>
                      <Button
                        variant="contained"
                        color="primary"
                        fullWidth
                        size="large"
                        onClick={handleUpload}
                        disabled={uploading || analyzing || !file || !caseName.trim()}
                        startIcon={<CloudUploadIcon />}
                        sx={{ 
                          borderRadius: 8,
                          py: 1.2,
                          background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                          textTransform: 'none',
                          fontSize: '1rem',
                          fontWeight: 500,
                          boxShadow: theme.shadows[4]
                        }}
                      >
                        Upload and Analyze
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Step 3: Progress Indicators */}
        {(uploading || analyzing) && (
          <Box sx={{
            ...itemInitial,
            ...fadeIn,
            transitionDelay: '0.2s'
          }}>
            <Card sx={{ 
              borderRadius: 2, 
              boxShadow: theme.shadows[3], 
              overflow: 'hidden',
              mb: 3
            }}>
              <CardContent>
                {/* Upload progress */}
                {uploading && (
                  <Box sx={{ mb: 4 }}>
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      mb: 1.5,
                      alignItems: 'center'
                    }}>
                      <Typography variant="subtitle1" fontWeight={500}>
                        Uploading
                      </Typography>
                      <Typography variant="body2" color={uploadProgress === 100 ? 'success.main' : 'text.secondary'}>
                        {Math.round(uploadProgress)}%
                      </Typography>
                    </Box>
                    <StyledProgress variant="determinate" value={uploadProgress} />
                    {uploadProgress === 100 && (
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <CheckCircleOutlineIcon sx={{ color: 'success.main', mr: 1, fontSize: 18 }} />
                        <Typography variant="caption" color="success.main">
                          Upload complete
                        </Typography>
                      </Box>
                    )}
                  </Box>
                )}

                {/* Analysis progress */}
                {analyzing && (
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      mb: 1.5,
                      alignItems: 'center'
                    }}>
                      <Typography variant="subtitle1" fontWeight={500}>
                        Analyzing Video
                      </Typography>
                      <Typography variant="body2" color={analysisProgress === 100 ? 'success.main' : 'text.secondary'}>
                        {Math.round(analysisProgress)}%
                      </Typography>
                    </Box>
                    <StyledProgress 
                      variant={analysisProgress > 0 ? "determinate" : "indeterminate"} 
                      value={analysisProgress} 
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Analysis may take several minutes depending on video length
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>

            {/* Redirect message */}
            <Alert 
              severity="info"
              icon={false}
              sx={{ 
                borderRadius: 2,
                backgroundColor: alpha(theme.palette.info.main, 0.08),
                color: theme.palette.info.dark
              }}
            >
              <Typography variant="body2">
                You'll be redirected to the cases page while analysis continues in the background.
                You can check progress there.
              </Typography>
            </Alert>
          </Box>
        )}

        {/* Success state */}
        {(success && caseId) && (
          <Box sx={{
            opacity: 0,
            transform: 'scale(0.9)',
            animation: 'fadeInScale 0.6s forwards',
            '@keyframes fadeInScale': {
              to: { opacity: 1, transform: 'scale(1)' }
            }
          }}>
            <Box sx={{ textAlign: 'center', my: 3 }}>
              <Box sx={{
                animation: 'popIn 0.5s forwards 0.3s',
                opacity: 0,
                transform: 'scale(0)',
                '@keyframes popIn': {
                  to: { opacity: 1, transform: 'scale(1)' }
                }
              }}>
                <CheckCircleOutlineIcon sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
              </Box>
              <Typography variant="h5" gutterBottom color="success.main" fontWeight={500}>
                Analysis Complete!
              </Typography>
              <Typography variant="body1" paragraph color="text.secondary">
                Your video has been successfully analyzed. View the results now.
              </Typography>
              
              <SuccessButton 
                variant="contained"
                size="large"
                onClick={() => navigate(`/cases/${caseId}`)}
                sx={{ 
                  mt: 2, 
                  px: 4, 
                  py: 1.5,
                  borderRadius: 8
                }}
              >
                View Analysis Results
              </SuccessButton>
            </Box>
          </Box>
        )}
      </UploadContainer>

      {/* Loading overlay */}
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={isLoading}
      >
        <CircularProgress color="inherit" />
      </Backdrop>
    </Box>
  );
};

export default UploadForm;