import React, { useEffect, useState, useRef } from 'react';
import { 
  Typography, Box, Paper, Grid, Tabs, Tab, CircularProgress, 
  Chip, Button, IconButton, Divider, List, ListItem, ListItemText,
  Dialog, DialogTitle, DialogContent, DialogActions, TextField,
  Card, CardContent, MenuItem, Menu, Tooltip, LinearProgress
} from '@mui/material';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';

// Icons
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import DownloadIcon from '@mui/icons-material/Download';
import ShareIcon from '@mui/icons-material/Share';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import BookmarkBorderIcon from '@mui/icons-material/BookmarkBorder';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import TimelineIcon from '@mui/icons-material/Timeline';
import PersonIcon from '@mui/icons-material/Person';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import InfoIcon from '@mui/icons-material/Info';
import FilterListIcon from '@mui/icons-material/FilterList';
import MoreVertIcon from '@mui/icons-material/MoreVert';

// Current date and time in YYYY-MM-DD HH:MM:SS format
const CURRENT_DATE = '2025-04-06 06:50:44';
const CURRENT_USER = 'JayanthVeerappa';

function CaseAnalysis() {
  const { caseName } = useParams(); // Use caseName from URL
  const navigate = useNavigate();
  
  // State
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [videoReady, setVideoReady] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [favorites, setFavorites] = useState(false);
  const [actionsMenuAnchor, setActionsMenuAnchor] = useState(null);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  
  // Refs
  const videoRef = useRef(null);
  
  // Form state for editing
  const [editForm, setEditForm] = useState({
    case_name: '',
    case_summary: '',
    case_priority: '',
    case_location: '',
    case_tags: ''
  });
  
  // Fetch case data
  useEffect(() => {
    const fetchCaseData = async () => {
      setLoading(true);
      try {
        // Fetch by case name instead of ID
        const response = await axios.get(`/api/cases/name/${encodeURIComponent(caseName)}`);
        setCaseData(response.data);
        
        // Initialize edit form with current values
        setEditForm({
          case_name: response.data.case_name,
          case_summary: response.data.case_summary || '',
          case_priority: response.data.case_priority || 'medium',
          case_location: response.data.case_location || '',
          case_tags: response.data.case_tags || ''
        });
        
        // Check if case is in favorites
        const storedFavorites = localStorage.getItem('favoritesCases') || '[]';
        const favoritesArray = JSON.parse(storedFavorites);
        setFavorites(favoritesArray.includes(response.data.case_id));
        
        setError(null);
      } catch (err) {
        console.error('Error fetching case data:', err);
        setError('Failed to load case data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    if (caseName) {
      fetchCaseData();
    }
  }, [caseName]);
  
  // Video event handlers
  const handleVideoReady = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      setVideoReady(true);
    }
  };
  
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };
  
  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  const handleSeek = (e) => {
    if (videoRef.current) {
      const newTime = (e.target.value / 100) * duration;
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };
  
  // Format time in MM:SS
  const formatTime = (timeInSeconds) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  
  // Tab handlers
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Menu handlers
  const openActionsMenu = (event) => {
    setActionsMenuAnchor(event.currentTarget);
  };
  
  const closeActionsMenu = () => {
    setActionsMenuAnchor(null);
  };
  
  // Favorites toggle
  const toggleFavorite = () => {
    if (!caseData) return;
    
    const storedFavorites = localStorage.getItem('favoritesCases') || '[]';
    let favoritesArray = JSON.parse(storedFavorites);
    
    if (favorites) {
      favoritesArray = favoritesArray.filter(id => id !== caseData.case_id);
    } else {
      favoritesArray.push(caseData.case_id);
    }
    
    localStorage.setItem('favoritesCases', JSON.stringify(favoritesArray));
    setFavorites(!favorites);
  };
  
  // Edit handlers
  const openEditDialog = () => {
    setEditDialogOpen(true);
    closeActionsMenu();
  };
  
  const handleEditChange = (field, value) => {
    setEditForm({
      ...editForm,
      [field]: value
    });
  };
  
  const handleEditSubmit = async () => {
    if (!caseData) return;
    
    try {
      const response = await axios.put(`/api/cases/${caseData.case_id}`, editForm);
      setCaseData({
        ...caseData,
        ...response.data
      });
      setEditDialogOpen(false);
      
      // If name changed, redirect to new URL
      if (response.data.case_name !== caseName) {
        navigate(`/cases/${encodeURIComponent(response.data.case_name)}`, { replace: true });
      }
    } catch (err) {
      console.error('Error updating case:', err);
      // Show error message
    }
  };
  
  // Delete handlers
  const openDeleteDialog = () => {
    setDeleteDialogOpen(true);
    closeActionsMenu();
  };
  
  const handleDeleteCase = async () => {
    if (!caseData) return;
    
    try {
      await axios.delete(`/api/cases/${caseData.case_id}`);
      setDeleteDialogOpen(false);
      navigate('/dashboard', { replace: true });
    } catch (err) {
      console.error('Error deleting case:', err);
      // Show error message
    }
  };
  
  // Share handlers
  const openShareDialog = () => {
    setShareDialogOpen(true);
    closeActionsMenu();
  };
  
  const handleShare = () => {
    // Implement share functionality
    setShareDialogOpen(false);
  };
  
  // Loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  // Error state
  if (error) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h5" color="error" gutterBottom>
          {error}
        </Typography>
        <Button variant="contained" onClick={() => navigate('/dashboard')}>
          Return to Dashboard
        </Button>
      </Box>
    );
  }
  
  // Not found state
  if (!caseData) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h5" color="error" gutterBottom>
          Case not found
        </Typography>
        <Typography paragraph>
          The case "{caseName}" could not be found. It may have been deleted or you may not have permission to view it.
        </Typography>
        <Button variant="contained" onClick={() => navigate('/dashboard')}>
          Return to Dashboard
        </Button>
      </Box>
    );
  }
  
  return (
    <Box sx={{ p: 3 }}>
      {/* Case Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography variant="h4">{caseData.case_name}</Typography> {/* Display case name */}
          
          <Chip 
            label={caseData.case_priority}
            color={
              caseData.case_priority === 'urgent' ? 'error' :
              caseData.case_priority === 'high' ? 'warning' :
              caseData.case_priority === 'medium' ? 'info' : 
              'default'
            }
            sx={{ ml: 2 }}
          />
          
          <Chip 
            label={caseData.status || 'new'}
            color={
              caseData.status === 'completed' ? 'success' :
              caseData.status === 'processing' ? 'warning' :
              'default'
            }
            sx={{ ml: 1 }}
          />
          
          <IconButton 
            color={favorites ? 'warning' : 'default'} 
            onClick={toggleFavorite}
            sx={{ ml: 1 }}
          >
            {favorites ? <BookmarkIcon /> : <BookmarkBorderIcon />}
          </IconButton>
        </Box>
        
        <Box>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            sx={{ mr: 1 }}
            onClick={() => {
              // Implement download functionality
              if (caseData.video_url) {
                window.open(caseData.video_url, '_blank');
              }
            }}
          >
            Download
          </Button>
          
          <IconButton 
            aria-label="more actions"
            aria-controls="case-actions-menu"
            aria-haspopup="true"
            onClick={openActionsMenu}
          >
            <MoreVertIcon />
          </IconButton>
          
          <Menu
            id="case-actions-menu"
            anchorEl={actionsMenuAnchor}
            keepMounted
            open={Boolean(actionsMenuAnchor)}
            onClose={closeActionsMenu}
          >
            <MenuItem onClick={openEditDialog}>
              <EditIcon fontSize="small" sx={{ mr: 1 }} />
              Edit Case Details
            </MenuItem>
            <MenuItem onClick={openShareDialog}>
              <ShareIcon fontSize="small" sx={{ mr: 1 }} />
              Share Case
            </MenuItem>
            <Divider />
            <MenuItem onClick={openDeleteDialog} sx={{ color: 'error.main' }}>
              <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
              Delete Case
            </MenuItem>
          </Menu>
        </Box>
      </Box>
      
      {/* Case Meta Information */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">Case Date</Typography>
            <Typography variant="body1">{new Date(caseData.case_date).toLocaleDateString()}</Typography>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">Case Time</Typography>
            <Typography variant="body1">{caseData.case_time || 'Not specified'}</Typography>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">Location</Typography>
            <Typography variant="body1">{caseData.case_location || 'Not specified'}</Typography>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">Submitted By</Typography>
            <Typography variant="body1">{caseData.submitted_by || 'Unknown'}</Typography>
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="body2" color="text.secondary">Summary</Typography>
            <Typography variant="body1">{caseData.case_summary || 'No summary provided'}</Typography>
          </Grid>
          
          {caseData.case_tags && (
            <Grid item xs={12}>
              <Typography variant="body2" color="text.secondary">Tags</Typography>
              <Box sx={{ mt: 1 }}>
                {caseData.case_tags.split(',').map((tag, index) => (
                  <Chip key={index} label={tag.trim()} size="small" sx={{ mr: 1, mb: 1 }} />
                ))}
              </Box>
            </Grid>
          )}
        </Grid>
      </Paper>
      
      {/* Video Player */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ position: 'relative', width: '100%', bgcolor: '#000', mb: 1 }}>
          <video
            ref={videoRef}
            style={{ width: '100%', display: 'block' }}
            src={caseData.video_url || ''}
            poster={caseData.thumbnail_url || ''}
            onLoadedMetadata={handleVideoReady}
            onTimeUpdate={handleTimeUpdate}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
          />
          
          {!videoReady && (
            <Box sx={{ 
              position: 'absolute', 
              top: 0, left: 0, right: 0, bottom: 0, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center' 
            }}>
              <CircularProgress color="secondary" />
            </Box>
          )}
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <IconButton onClick={togglePlayPause} disabled={!videoReady}>
            {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
          </IconButton>
          
          <Typography variant="body2" sx={{ minWidth: '70px' }}>
            {formatTime(currentTime)} / {formatTime(duration)}
          </Typography>
          
          <Box sx={{ flex: 1, mx: 2 }}>
            <input
              type="range"
              min="0"
              max="100"
              value={(currentTime / duration) * 100 || 0}
              onChange={handleSeek}
              disabled={!videoReady}
              style={{ width: '100%' }}
            />
          </Box>
        </Box>
        
        {/* Video analysis progress */}
        {caseData.status === 'processing' && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" gutterBottom>
              Analysis Progress: {caseData.analysis_progress || 0}%
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={caseData.analysis_progress || 0} 
              sx={{ height: 8, borderRadius: 1 }}
            />
          </Box>
        )}
      </Paper>
      
      {/* Analysis Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="analysis tabs">
          <Tab icon={<InfoIcon />} label="Overview" id="tab-0" aria-controls="tabpanel-0" />
          <Tab icon={<TimelineIcon />} label="Timeline" id="tab-1" aria-controls="tabpanel-1" />
          <Tab icon={<PersonIcon />} label="People" id="tab-2" aria-controls="tabpanel-2" />
          <Tab icon={<ViewInArIcon />} label="3D Scene" id="tab-3" aria-controls="tabpanel-3" />
        </Tabs>
      </Box>
      
      {/* Tab Content */}
      <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ py: 3 }}>
        {activeTab === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Analysis Summary</Typography>
                  <Typography variant="body2" paragraph>
                    {caseData.analysis_summary || 
                      'Analysis information will appear here once processing is complete.'}
                  </Typography>
                  
                  {caseData.status === 'completed' && (
                    <Box>
                      <Typography variant="body2">Key Findings:</Typography>
                      <List dense>
                        <ListItem>
                          <ListItemText 
                            primary="People Detected" 
                            secondary={caseData.people_count || '0'} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Objects Tracked" 
                            secondary={caseData.object_count || '0'} 
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemText 
                            primary="Events Identified" 
                            secondary={caseData.event_count || '0'} 
                          />
                        </ListItem>
                      </List>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>Processing Information</Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="Analysis Started" 
                        secondary={caseData.analysis_start_time ? new Date(caseData.analysis_start_time).toLocaleString() : 'Not started'} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Analysis Completed" 
                        secondary={caseData.analysis_end_time ? new Date(caseData.analysis_end_time).toLocaleString() : 'In progress'} 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Processing Time" 
                        secondary={
                          caseData.analysis_start_time && caseData.analysis_end_time 
                            ? `${Math.round((new Date(caseData.analysis_end_time) - new Date(caseData.analysis_start_time)) / 60000)} minutes` 
                            : 'Pending'
                        } 
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Models Used" 
                        secondary={caseData.models_used || 'Standard detection models'} 
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ py: 3 }}>
        {activeTab === 1 && (
          <Box>
            <Typography variant="h6" gutterBottom>Video Timeline</Typography>
            {caseData.status !== 'completed' ? (
              <Typography>Timeline will be available once analysis is complete.</Typography>
            ) : (
              <Box>
                {/* Timeline implementation would go here */}
                <Typography>Timeline visualization of events detected in the video.</Typography>
              </Box>
            )}
          </Box>
        )}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ py: 3 }}>
        {activeTab === 2 && (
          <Box>
            <Typography variant="h6" gutterBottom>People Detected</Typography>
            {caseData.status !== 'completed' ? (
              <Typography>People detection results will be available once analysis is complete.</Typography>
            ) : (
              <Grid container spacing={2}>
                {/* Person cards would go here */}
                <Grid item xs={12} sm={6} md={4}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle1">Person #1</Typography>
                      <Typography variant="body2">
                        Appeared from 00:12 to 01:45
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}
          </Box>
        )}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 3} id="tabpanel-3" aria-labelledby="tab-3" sx={{ py: 3 }}>
        {activeTab === 3 && (
          <Box>
            <Typography variant="h6" gutterBottom>3D Scene Reconstruction</Typography>
            {caseData.status !== 'completed' ? (
              <Typography>3D scene reconstruction will be available once analysis is complete.</Typography>
            ) : (
              <Box>
                {/* 3D scene visualization would go here */}
                <Typography>3D visualization of the scene extracted from the video.</Typography>
              </Box>
            )}
          </Box>
        )}
      </Box>
      
      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} fullWidth maxWidth="md">
        <DialogTitle>Edit Case Details</DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                label="Case Name"
                fullWidth
                required
                value={editForm.case_name}
                onChange={(e) => handleEditChange('case_name', e.target.value)}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                label="Location"
                fullWidth
                value={editForm.case_location}
                onChange={(e) => handleEditChange('case_location', e.target.value)}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                select
                label="Priority"
                fullWidth
                value={editForm.case_priority}
                onChange={(e) => handleEditChange('case_priority', e.target.value)}
              >
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="urgent">Urgent</MenuItem>
              </TextField>
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                label="Tags"
                fullWidth
                value={editForm.case_tags}
                onChange={(e) => handleEditChange('case_tags', e.target.value)}
                helperText="Comma-separated keywords"
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                label="Summary"
                fullWidth
                multiline
                rows={4}
                value={editForm.case_summary}
                onChange={(e) => handleEditChange('case_summary', e.target.value)}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleEditSubmit}
            variant="contained" 
            color="primary"
            disabled={!editForm.case_name.trim()}
          >
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Share Dialog */}
      <Dialog open={shareDialogOpen} onClose={() => setShareDialogOpen(false)}>
        <DialogTitle>Share Case</DialogTitle>
        <DialogContent dividers>
          <Typography paragraph>
            Share access to this case with other users:
          </Typography>
          <TextField
            label="Email or username"
            fullWidth
            sx={{ mb: 2 }}
          />
          <TextField
            select
            label="Permission level"
            fullWidth
            defaultValue="view"
          >
            <MenuItem value="view">View only</MenuItem>
            <MenuItem value="edit">Can edit</MenuItem>
            <MenuItem value="admin">Administrator</MenuItem>
          </TextField>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShareDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleShare} variant="contained" color="primary">
            Share
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Case</DialogTitle>
        <DialogContent dividers>
          <Typography paragraph>
            Are you sure you want to delete case "{caseData.case_name}"? This action cannot be undone.
          </Typography>
          <Typography color="error">
            All associated videos, analysis results, and annotations will be permanently deleted.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteCase} variant="contained" color="error">
            Delete Permanently
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Footer */}
      <Box sx={{ mt: 4, textAlign: 'right', color: 'text.secondary' }}>
        <Typography variant="caption">
          Last updated: {caseData.last_updated || CURRENT_DATE} • User: {CURRENT_USER}
        </Typography>
      </Box>
    </Box>
  );
}

export default CaseAnalysis;