import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Box, Typography, Paper, Grid, Card, CardContent, 
  List, ListItem, ListItemText, ListItemIcon, Divider,
  IconButton, Tooltip, Button, Chip, CircularProgress,
  Alert, Slider, Stack, styled
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import PersonIcon from '@mui/icons-material/Person';
import TimelineIcon from '@mui/icons-material/Timeline';
import MapIcon from '@mui/icons-material/Map';
import ViewListIcon from '@mui/icons-material/ViewList';
import ShareIcon from '@mui/icons-material/Share';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import VisibilityIcon from '@mui/icons-material/Visibility';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import VolumeOffIcon from '@mui/icons-material/VolumeOff';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import axios from 'axios';

// Current date and user info from latest request
const CURRENT_DATE = '2025-04-06 08:44:28';
const CURRENT_USER = 'aaravgoel0';

// Styled components for custom video player UI
const VideoContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  backgroundColor: '#000',
  '&:hover .video-controls': {
    opacity: 1,
  },
}));

const VideoControls = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: 0,
  left: 0,
  right: 0,
  background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%)',
  padding: theme.spacing(1, 2),
  display: 'flex',
  flexDirection: 'column',
  opacity: 0,
  transition: 'opacity 0.3s ease',
  color: 'white',
  zIndex: 5
}));

const ProgressSlider = styled(Slider)(({ theme }) => ({
  color: theme.palette.primary.main,
  height: 4,
  '& .MuiSlider-thumb': {
    width: 12,
    height: 12,
    display: 'none',
  },
  '&:hover .MuiSlider-thumb': {
    display: 'block',
  },
  padding: '15px 0'
}));

const VideoPlayer = () => {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const [caseData, setCaseData] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.7);
  const [muted, setMuted] = useState(false);
  const [relatedEvents, setRelatedEvents] = useState([]);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [showControls, setShowControls] = useState(false);
  const [fallbackMode, setFallbackMode] = useState(false);

  useEffect(() => {
    // Fetch case data and video URL
    const fetchCaseData = async () => {
      try {
        setLoading(true);
        
        // Get case details
        const response = await axios.get(`/api/cases/${caseId}`);
        setCaseData(response.data);
        
        console.log('Case data received:', response.data);
        
        // Check if output_video exists
        if (response.data?.output_video && typeof response.data.output_video === 'string') {
          console.log('Found video URL:', response.data.output_video);
          setVideoUrl(response.data.output_video);
        } else {
          console.log('No video URL in response, using default path');
          setVideoUrl(`/api/videos/${caseId}`);
        }
        
        try {
          // Try to fetch related events, but don't fail if endpoint doesn't exist
          const eventsResponse = await axios.get(`/api/cases/${caseId}/events`);
          if (eventsResponse.data) {
            setRelatedEvents(eventsResponse.data);
          }
        } catch (eventErr) {
          console.log('Events API not available, continuing without events data');
          setRelatedEvents([]);
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching case data:', err);
        setError('Failed to load case data. Please try again.');
        setLoading(false);
      }
    };

    fetchCaseData();
  }, [caseId]);

  // Video playback control functions
  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleVolumeChange = (event, newValue) => {
    setVolume(newValue);
    if (videoRef.current) {
      videoRef.current.volume = newValue;
      setMuted(newValue === 0);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !muted;
      setMuted(!muted);
    }
  };

  const handleProgressChange = (event, newValue) => {
    if (videoRef.current) {
      videoRef.current.currentTime = newValue;
      setCurrentTime(newValue);
    }
  };

  const handlePlaybackSpeedChange = (speed) => {
    setPlaybackSpeed(speed);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  };

  const handleFullscreen = () => {
    if (containerRef.current) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen();
      } else if (containerRef.current.webkitRequestFullscreen) {
        containerRef.current.webkitRequestFullscreen();
      } else if (containerRef.current.msRequestFullscreen) {
        containerRef.current.msRequestFullscreen();
      }
    }
  };

  const jumpToEvent = (timestamp) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timestamp;
      setCurrentTime(timestamp);
      if (!isPlaying) {
        videoRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  const handleBack = () => {
    navigate(`/cases/${caseId}`);
  };

  const handleVideoError = () => {
    console.error('Video playback error');
    setError('Error playing video. Trying alternative playback method...');
    setFallbackMode(true);
  };

  // Filter events that are close to the current playback time
  const currentEvents = relatedEvents.filter(
    event => Math.abs(event.timestamp - currentTime) < 2
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>Loading video player...</Typography>
      </Box>
    );
  }

  // Get the filename portion for display
  const videoFilename = videoUrl?.split('/').pop() || 'video';

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Button startIcon={<ArrowBackIcon />} onClick={handleBack}>
          Back to Case
        </Button>
        <Box>
          <Chip 
            label={`Case: ${caseId}`} 
            color="primary" 
            sx={{ mr: 1 }} 
          />
          <Chip 
            icon={<PersonIcon />} 
            label={`Viewed by: ${CURRENT_USER}`} 
            variant="outlined" 
            size="small" 
          />
        </Box>
      </Box>

      {error && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Video Player */}
        <Grid item xs={12} md={8}>
          <Paper 
            elevation={5} 
            sx={{ 
              borderRadius: 2, 
              overflow: 'hidden', 
              bgcolor: '#000',
              position: 'relative' 
            }}
          >
            {/* Custom Video Player */}
            {videoUrl && !fallbackMode && (
              <VideoContainer 
                ref={containerRef}
                onMouseEnter={() => setShowControls(true)}
                onMouseLeave={() => setShowControls(false)}
              >
                <video
                  ref={videoRef}
                  width="100%"
                  style={{ display: 'block' }}
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onError={handleVideoError}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                  preload="auto"
                >
                  <source src={videoUrl} type="video/mp4" />
                  <source src={videoUrl} type="video/webm" />
                  <source src={videoUrl} type="video/ogg" />
                  Your browser does not support HTML5 video.
                </video>

                {/* Custom Controls */}
                <VideoControls className="video-controls" style={{ opacity: showControls ? 1 : 0 }}>
                  <ProgressSlider
                    value={currentTime}
                    max={duration || 100}
                    onChange={handleProgressChange}
                    aria-label="video progress"
                  />
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <IconButton 
                        onClick={togglePlay} 
                        size="small" 
                        sx={{ color: 'white', mr: 1 }}
                      >
                        {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                      </IconButton>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', width: 150, mr: 2 }}>
                        <IconButton 
                          onClick={toggleMute} 
                          size="small" 
                          sx={{ color: 'white' }}
                        >
                          {muted ? <VolumeOffIcon /> : <VolumeUpIcon />}
                        </IconButton>
                        <Slider
                          size="small"
                          value={muted ? 0 : volume}
                          onChange={handleVolumeChange}
                          min={0}
                          max={1}
                          step={0.1}
                          sx={{ 
                            ml: 1, 
                            color: 'white',
                            '& .MuiSlider-thumb': {
                              width: 10,
                              height: 10,
                            }
                          }}
                        />
                      </Box>
                      
                      <Typography variant="body2" sx={{ color: 'white' }}>
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box sx={{ mr: 2 }}>
                        <Button 
                          variant="text" 
                          size="small"
                          onClick={() => handlePlaybackSpeedChange(0.5)}
                          sx={{ 
                            color: 'white', 
                            minWidth: 'auto', 
                            px: 1,
                            opacity: playbackSpeed === 0.5 ? 1 : 0.7,
                            bgcolor: playbackSpeed === 0.5 ? 'rgba(255,255,255,0.2)' : 'transparent'
                          }}
                        >
                          0.5x
                        </Button>
                        <Button 
                          variant="text" 
                          size="small"
                          onClick={() => handlePlaybackSpeedChange(1)}
                          sx={{ 
                            color: 'white', 
                            minWidth: 'auto', 
                            px: 1,
                            opacity: playbackSpeed === 1 ? 1 : 0.7,
                            bgcolor: playbackSpeed === 1 ? 'rgba(255,255,255,0.2)' : 'transparent'
                          }}
                        >
                          1x
                        </Button>
                        <Button 
                          variant="text" 
                          size="small"
                          onClick={() => handlePlaybackSpeedChange(2)}
                          sx={{ 
                            color: 'white', 
                            minWidth: 'auto', 
                            px: 1,
                            opacity: playbackSpeed === 2 ? 1 : 0.7,
                            bgcolor: playbackSpeed === 2 ? 'rgba(255,255,255,0.2)' : 'transparent'
                          }}
                        >
                          2x
                        </Button>
                      </Box>
                      
                      <IconButton 
                        onClick={handleFullscreen} 
                        size="small" 
                        sx={{ color: 'white' }}
                      >
                        <FullscreenIcon />
                      </IconButton>
                    </Box>
                  </Box>
                </VideoControls>
              </VideoContainer>
            )}
            
            {/* Fallback QuickTime/HTML5 Embed */}
            {videoUrl && fallbackMode && (
              <Box sx={{ p: 2, bgcolor: '#000', borderRadius: 0, textAlign: 'center' }}>
                <Typography variant="body2" color="white" gutterBottom>
                  Using alternative player for {videoFilename}
                </Typography>
                
                {/* QuickTime Player Fallback */}
                <object
                  type="video/quicktime"
                  data={videoUrl}
                  width="100%"
                  height="400"
                  style={{ border: 'none' }}
                >
                  <param name="controller" value="true" />
                  <param name="autoplay" value="true" />
                  <param name="scale" value="aspect" />
                  
                  {/* Additional fallback for browsers without QuickTime */}
                  <embed 
                    src={videoUrl}
                    width="100%" 
                    height="400" 
                    controller="true" 
                    autoplay="true"
                    scale="aspect"
                    pluginspage="http://www.apple.com/quicktime/download/"
                  />
                  
                  <Typography variant="body2" color="white" sx={{ mt: 2 }}>
                    Your browser doesn't support embedded videos. 
                    <br/>
                    <Button
                      variant="outlined"
                      color="primary"
                      href={videoUrl}
                      target="_blank"
                      size="small"
                      sx={{ mt: 1, color: 'white', borderColor: 'white' }}
                    >
                      Download Video
                    </Button>
                  </Typography>
                </object>
              </Box>
            )}
            
            {/* No Video URL */}
            {!videoUrl && (
              <Box 
                sx={{ 
                  width: '100%', 
                  height: '400px', 
                  display: 'flex', 
                  justifyContent: 'center', 
                  alignItems: 'center',
                  bgcolor: '#333',
                  color: 'white',
                  flexDirection: 'column'
                }}
              >
                <Typography variant="h6" gutterBottom>Video unavailable</Typography>
                <Typography variant="body2">The video for this case could not be loaded</Typography>
              </Box>
            )}
            
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              bgcolor: '#111', 
              color: 'white', 
              p: 1.5,
              borderTop: '1px solid rgba(255,255,255,0.1)'
            }}>
              <Typography variant="body2">
                {videoFilename}
              </Typography>
              <Typography variant="body2">
                Analysis: {CURRENT_DATE}
              </Typography>
            </Box>
          </Paper>

          {/* Current Events */}
          {currentEvents.length > 0 && (
            <Paper sx={{ p: 2, mt: 2, borderRadius: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Currently Visible:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {currentEvents.map((event, index) => (
                  <Chip 
                    key={index}
                    label={event.label || 'Object'}
                    color="primary"
                    variant="outlined"
                    icon={<VisibilityIcon />}
                    onClick={() => navigate(`/cases/${caseId}/objects/${event.id}`)}
                  />
                ))}
              </Box>
            </Paper>
          )}

          {/* Video Details */}
          <Paper sx={{ p: 3, mt: 3, borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>
              Video Details
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  Case Name
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {caseData?.name || caseId}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Typography variant="body2" color="text.secondary">
                  Upload Date
                </Typography>
                <Typography variant="body1" gutterBottom>
                  {caseData?.timestamp ? new Date(caseData.timestamp).toLocaleString() : CURRENT_DATE}
                </Typography>
              </Grid>
              {caseData?.video_path && (
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">
                    Original Video
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {caseData.video_path.split('/').pop()}
                  </Typography>
                </Grid>
              )}
              <Grid item xs={12}>
                <Typography variant="body2" color="text.secondary">
                  Viewing Session
                </Typography>
                <Typography variant="body1" gutterBottom>
                  Viewed by {CURRENT_USER} on {CURRENT_DATE}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Sidebar */}
        <Grid item xs={12} md={4}>
          {/* Related Analysis */}
          <Card sx={{ mb: 3, borderRadius: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Connected Analysis
              </Typography>
              <List dense>
                <ListItem 
                  button 
                  onClick={() => navigate(`/cases/${caseId}/timeline`)}
                >
                  <ListItemIcon>
                    <TimelineIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Timeline Analysis" 
                    secondary="View events across time"
                  />
                </ListItem>
                <Divider />
                <ListItem 
                  button 
                  onClick={() => navigate(`/cases/${caseId}/persons`)}
                >
                  <ListItemIcon>
                    <PersonIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Person Tracking" 
                    secondary="Identified individuals"
                  />
                </ListItem>
                <Divider />
                <ListItem 
                  button 
                  onClick={() => navigate(`/cases/${caseId}/map`)}
                >
                  <ListItemIcon>
                    <MapIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Spatial Mapping" 
                    secondary="3D reconstruction"
                  />
                </ListItem>
                <Divider />
                <ListItem 
                  button 
                  onClick={() => navigate(`/cases/${caseId}/objects`)}
                >
                  <ListItemIcon>
                    <ViewListIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Object Inventory" 
                    secondary="All detected objects"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>

          {/* Events Timeline */}
          <Card sx={{ borderRadius: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Key Events
              </Typography>
              {relatedEvents.length > 0 ? (
                <List dense>
                  {relatedEvents.slice(0, 10).map((event, index) => (
                    <React.Fragment key={index}>
                      <ListItem 
                        button 
                        onClick={() => jumpToEvent(event.timestamp)}
                      >
                        <ListItemText 
                          primary={event.label || 'Event'}
                          secondary={`${formatTime(event.timestamp)} - ${event.description || 'Detected object'}`}
                        />
                      </ListItem>
                      {index < relatedEvents.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Events data isn't available for this video. You can still view the timeline analysis for detailed event information.
                </Typography>
              )}
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
            <Tooltip title="Share this video">
              <IconButton>
                <ShareIcon />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Save bookmark">
              <IconButton>
                <BookmarkIcon />
              </IconButton>
            </Tooltip>
            
            <Button 
              variant="contained" 
              color="primary"
              onClick={() => navigate(`/cases/${caseId}/chat`)}
            >
              Ask About This Video
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default VideoPlayer;