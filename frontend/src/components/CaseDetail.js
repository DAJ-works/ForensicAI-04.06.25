import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { Typography, Box, Grid, Paper, Tabs, Tab, Button, CircularProgress, Chip, Divider, Tooltip, Snackbar, Alert } from '@mui/material';
import PersonGrid from './PersonGrid';
import axios from 'axios';
import VideocamIcon from '@mui/icons-material/Videocam';
import DownloadIcon from '@mui/icons-material/Download';
import TimelineIcon from '@mui/icons-material/Timeline';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import MapIcon from '@mui/icons-material/Map';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';

function CaseDetail() {
  const navigate = useNavigate();
  const { caseId } = useParams();
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [persons, setPersons] = useState([]);
  const [downloadingVideo, setDownloadingVideo] = useState(false);
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('info');

  // Current user and timestamp for logging
  const currentUser = "aaravgoel0";
  const currentDate = "2025-04-06 09:01:50";

  useEffect(() => {
    async function fetchCaseData() {
      try {
        setLoading(true);
        
        // Get case details
        const response = await axios.get(`/api/cases/${caseId}`);
        setCaseData(response.data);
        
        // Get persons
        const personsResponse = await axios.get(`/api/cases/${caseId}/persons`);
        setPersons(personsResponse.data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching case details:', err);
        setError('Failed to load case data');
        setLoading(false);
      }
    }
    
    fetchCaseData();
  }, [caseId]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleDownloadVideo = async () => {
    if (!caseData?.output_video) return;
    
    try {
      setDownloadingVideo(true);
      
      // Use the dedicated download endpoint
      const downloadUrl = `/api/cases/${caseId}/download_video?user=aaravgoel0&t=${Date.now()}`;
      
      // Use fetch API to handle the download as a blob
      const response = await fetch(downloadUrl);
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      // Get the filename from the Content-Disposition header if available
      let filename = `analyzed_video_${caseId}.mp4`;
      const contentDisposition = response.headers.get('Content-Disposition');
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=(.+)/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1].replace(/["']/g, '');
        }
      }
      
      // Convert response to blob
      const blob = await response.blob();
      
      // Create object URL for the blob
      const url = window.URL.createObjectURL(blob);
      
      // Create download link
      const link = document.createElement('a');
      link.href = url;
      link.download = filename; // Set download attribute to force download
      
      // Append to document, click, and remove
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the object URL
      window.URL.revokeObjectURL(url);
      
      console.log(`Video download completed for case: ${caseId}`);
      
    } catch (err) {
      console.error('Error downloading video:', err);
      alert('Failed to download video. Please try again later.');
    } finally {
      setTimeout(() => {
        setDownloadingVideo(false);
      }, 1000);
    }
  };

  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setShowSnackbar(false);
  };

  if (loading || error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!caseData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <Typography>Case not found</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Case Analysis: {caseId}
        </Typography>
        <Box>
          <Button 
            component={Link} 
            to="/cases" 
            variant="outlined" 
            color="primary" 
            sx={{ mr: 2 }}
          >
            Back to Cases
          </Button>
          <Button 
            component={Link} 
            to={`/cases/${caseId}/timeline`} 
            variant="contained" 
            color="primary" 
            startIcon={<TimelineIcon />}
          >
            Timeline View
          </Button>
        </Box>
      </Box>
      
      {/* Case Summary */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              {caseData.video_path?.split('/').pop() || 'Video Analysis'}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Analysis timestamp: {new Date(caseData.timestamp).toLocaleString()}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Total detections: {caseData.total_detections}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Unique objects: {caseData.total_unique_objects}
            </Typography>
            <Typography variant="body2" gutterBottom>
              Unique persons: {persons.length}
            </Typography>
            
            {/* Display any 3D reconstruction data if available */}
            {caseData['3d_analysis'] && caseData['3d_analysis'].reconstruction_available && (
              <Box sx={{ mt: 2 }}>
                <Chip 
                  icon={<ViewInArIcon />} 
                  label="3D Reconstruction Available" 
                  color="success" 
                  variant="outlined" 
                />
              </Box>
            )}
            
            {/* Video Player Button */}
            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<VideoLibraryIcon />}
                onClick={() => navigate(`/cases/${caseId}/video`)}
                fullWidth
                sx={{ 
                  py: 1.5,
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    transition: 'transform 0.3s'
                  }
                }}
              >
                View Video in Interactive Player
              </Button>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            {caseData.output_video && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                {/* Download button */}
                <Tooltip title="Download the analyzed video to your device">
                  <Button 
                    variant="contained" 
                    color="secondary" 
                    startIcon={downloadingVideo ? <CircularProgress size={20} color="inherit" /> : <DownloadIcon />}
                    onClick={handleDownloadVideo}
                    disabled={downloadingVideo}
                    sx={{ 
                      mx: 1,
                      position: 'relative',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        transition: 'transform 0.3s'
                      }
                    }}
                  >
                    {downloadingVideo ? 'Downloading...' : 'Download Analyzed Video'}
                  </Button>
                </Tooltip>
                
                {/* View button with purple outline - REMOVED */}
                
                {caseData.visualizations?.trajectories && (
                  <Button 
                    variant="contained" 
                    color="primary" 
                    startIcon={<MapIcon />}
                    href={caseData.visualizations.trajectories}
                    target="_blank"
                    sx={{ mx: 1 }}
                  >
                    View Trajectories
                  </Button>
                )}
              </Box>
            )}
          </Grid>
        </Grid>
      </Paper>
      
      {/* Visualizations */}
      {caseData.visualizations && Object.keys(caseData.visualizations).length > 0 && (
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>Visualizations</Typography>
          
          <Grid container spacing={3}>
            {caseData.visualizations.person_map && (
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle1">Person Position Map</Typography>
                </Box>
                <img 
                  src={caseData.visualizations.person_map} 
                  alt="Person Map" 
                  style={{ width: '100%', height: 'auto', maxHeight: '400px', objectFit: 'contain', border: '1px solid #ddd' }}
                />
              </Grid>
            )}
            
            {caseData.visualizations.reid_gallery && (
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle1">Person Gallery</Typography>
                </Box>
                <img 
                  src={caseData.visualizations.reid_gallery} 
                  alt="Person Gallery" 
                  style={{ width: '100%', height: 'auto', maxHeight: '400px', objectFit: 'contain', border: '1px solid #ddd' }}
                />
              </Grid>
            )}
            
            {caseData.visualizations.timeline && (
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle1">Event Timeline</Typography>
                </Box>
                <img 
                  src={caseData.visualizations.timeline} 
                  alt="Event Timeline" 
                  style={{ width: '100%', height: 'auto', maxHeight: '400px', objectFit: 'contain', border: '1px solid #ddd' }}
                />
              </Grid>
            )}
            
            {caseData.visualizations.heatmap && (
              <Grid item xs={12} md={6}>
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle1">Movement Heatmap</Typography>
                </Box>
                <img 
                  src={caseData.visualizations.heatmap} 
                  alt="Movement Heatmap" 
                  style={{ width: '100%', height: 'auto', maxHeight: '400px', objectFit: 'contain', border: '1px solid #ddd' }}
                />
              </Grid>
            )}
          </Grid>
        </Paper>
      )}
      
      {/* Tabs for different data views - RESTORED OBJECT DETECTIONS TAB */}
      <Paper elevation={3} sx={{ p: 3 }}>
        <Box sx={{ width: '100%' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} aria-label="case data tabs">
              <Tab label="Persons" id="tab-0" aria-controls="tabpanel-0" />
              <Tab label="Object Detections" id="tab-1" aria-controls="tabpanel-1" />
            </Tabs>
          </Box>
          
          {/* Persons Tab */}
          <TabPanel value={activeTab} index={0}>
            <Typography variant="h5" gutterBottom>Identified Persons</Typography>
            {persons.length === 0 ? (
              <Box sx={{ mt: 2, mb: 4 }}>
                <Typography variant="body1" color="error">
                  No persons detected in this case.
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  This may be due to:
                  <ul>
                    <li>Low confidence detections (try adjusting confidence threshold)</li>
                    <li>Low video quality or poor lighting</li>
                    <li>No people present in the video</li>
                    <li>Processing error (check logs for details)</li>
                  </ul>
                </Typography>
                
                {caseData.debug_info && (
                  <Paper elevation={1} sx={{ p: 2, mt: 2, bgcolor: '#f8f8f8' }}>
                    <Typography variant="subtitle2">Debug Information:</Typography>
                    <pre style={{ background: '#f0f0f0', padding: '10px', borderRadius: '4px', overflowX: 'auto', fontSize: '12px' }}>
                      {JSON.stringify(caseData.debug_info, null, 2)}
                    </pre>
                  </Paper>
                )}
                
                <Box sx={{ mt: 3 }}>
                  <Button 
                    variant="contained" 
                    color="primary"
                    onClick={() => window.location.reload()}
                  >
                    Reload Data
                  </Button>
                </Box>
              </Box>
            ) : (
              <PersonGrid persons={persons} caseId={caseId} />
            )}
          </TabPanel>
          
          {/* Object Detections Tab - RESTORED */}
          <TabPanel value={activeTab} index={1}>
            <Typography variant="h5" gutterBottom>Object Detections</Typography>
            
            <Grid container spacing={3}>
              {Object.entries(caseData.class_counts || {}).map(([className, count], index) => (
                <Grid item key={index} xs={6} sm={4} md={3} lg={2}>
                  <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h6">{count}</Typography>
                    <Typography variant="body2" color="text.secondary">{className}</Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
            
            {/* Display tracks if available */}
            {caseData.tracks && caseData.tracks.length > 0 && (
              <Box sx={{ mt: 4 }}>
                <Typography variant="h6" gutterBottom>Object Tracks</Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Showing top {Math.min(caseData.tracks.length, 10)} tracks by duration
                </Typography>
                
                <Grid container spacing={2}>
                  {caseData.tracks.slice(0, 10).map((track, index) => (
                    <Grid item key={index} xs={12} sm={6} md={4}>
                      <Paper elevation={2} sx={{ p: 2 }}>
                        <Typography variant="subtitle1">
                          {track.class_name} #{track.object_id}
                        </Typography>
                        <Typography variant="body2">
                          Duration: {track.duration?.toFixed(2) || 'unknown'} seconds
                        </Typography>
                        <Typography variant="body2">
                          Frames: {track.first_frame} - {track.last_frame}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
          </TabPanel>
        </Box>
      </Paper>
      
      {/* Snackbar for notifications */}
      <Snackbar 
        open={showSnackbar} 
        autoHideDuration={6000} 
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
}

// Tab Panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
      style={{ padding: '20px 0' }}
    >
      {value === index && (
        <Box>
          {children}
        </Box>
      )}
    </div>
  );
}

export default CaseDetail;