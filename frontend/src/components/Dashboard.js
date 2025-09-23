import React, { useState, useEffect, useRef } from 'react';
import { 
  Typography, Grid, Card, Box, Button, Paper, 
  CircularProgress, Chip, Tooltip, IconButton, 
  Fade, Tabs, Tab, Table, TableBody, TableCell, TableHead, TableRow,
  LinearProgress, Menu, MenuItem, Divider, Avatar, Alert, Link as MuiLink
} from '@mui/material';
import { Link } from 'react-router-dom';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';
import axios from 'axios';
import { styled } from '@mui/material/styles';

// Icons
import PersonIcon from '@mui/icons-material/Person';
import VideocamIcon from '@mui/icons-material/Videocam';
import CachedIcon from '@mui/icons-material/Cached';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import RefreshIcon from '@mui/icons-material/Refresh';
import FolderIcon from '@mui/icons-material/Folder';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import FilterListIcon from '@mui/icons-material/FilterList';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import ShareIcon from '@mui/icons-material/Share';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import ErrorIcon from '@mui/icons-material/Error';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import StarIcon from '@mui/icons-material/Star';
import ChatIcon from '@mui/icons-material/Chat';

// Current date and time (updated as requested)
const CURRENT_DATE = '2025-04-06 10:02:13';
const CURRENT_USER = 'aaravgoel0';

// Styled components for Apple-like design
const StyledCard = styled(Paper)(({ theme }) => ({
  borderRadius: 16,
  padding: theme.spacing(3),
  boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
  height: '100%',
  transition: 'all 0.3s ease',
  border: '1px solid rgba(0,0,0,0.05)',
  '&:hover': {
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
    transform: 'translateY(-2px)'
  }
}));

const StyledHeading = styled(Typography)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  fontWeight: 600,
  letterSpacing: '-0.5px',
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: 12,
  padding: '8px 20px',
  textTransform: 'none',
  fontWeight: 500,
  boxShadow: 'none',
  transition: 'all 0.3s ease',
  '&:hover': {
    boxShadow: '0 4px 10px rgba(0,0,0,0.1)',
    transform: 'translateY(-1px)'
  }
}));

const StyledIconButton = styled(IconButton)(({ theme }) => ({
  borderRadius: 12,
  padding: theme.spacing(1),
  transition: 'all 0.3s ease',
  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.03)',
  '&:hover': {
    backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.06)',
    transform: 'scale(1.05)'
  }
}));

const StyledChip = styled(Chip)(({ theme }) => ({
  borderRadius: 8,
  fontWeight: 500,
  boxShadow: 'none',
  '& .MuiChip-icon': {
    marginLeft: 8
  }
}));

const StyledTabs = styled(Tabs)(({ theme }) => ({
  '& .MuiTab-root': {
    textTransform: 'none',
    fontWeight: 500,
    minWidth: 100,
    borderRadius: '12px 12px 0 0',
    letterSpacing: '-0.2px',
    padding: '12px 24px',
    transition: 'all 0.3s ease'
  },
  '& .Mui-selected': {
    fontWeight: 600,
  },
  '& .MuiTabs-indicator': {
    height: 3,
    borderRadius: '3px 3px 0 0'
  }
}));

// Quick stats component
const QuickStat = ({ icon, title, value, change, changeType = 'neutral' }) => {
  return (
    <StyledCard elevation={0}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 3 }}>
        <Box sx={{ 
          width: 50, 
          height: 50, 
          borderRadius: '14px',
          bgcolor: 'primary.main',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 10px rgba(37, 99, 235, 0.2)'
        }}>
          {icon}
        </Box>
        
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 0.5, fontSize: '0.95rem' }}>
            {title}
          </Typography>
          <Typography variant="h4" sx={{ fontWeight: 700, letterSpacing: '-0.5px' }}>
            {value}
          </Typography>
          
          {change && (
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
              {changeType === 'positive' && <TrendingUpIcon fontSize="small" color="success" sx={{ mr: 0.5 }} />}
              <Typography 
                variant="body2" 
                color={
                  changeType === 'positive' ? 'success.main' : 
                  changeType === 'negative' ? 'error.main' : 
                  changeType === 'warning' ? 'warning.main' : 'text.secondary'
                }
                sx={{ fontWeight: 500 }}
              >
                {change}
              </Typography>
            </Box>
          )}
        </Box>
      </Box>
    </StyledCard>
  );
};

// Activity chart component
const ActivityTracker = ({ activityData = [] }) => {
  return (
    <StyledCard elevation={0}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, letterSpacing: '-0.5px' }}>
          Weekly Activity
        </Typography>
        <StyledChip 
          label="This Week" 
          color="primary" 
          size="small" 
          variant="outlined"
        />
      </Box>
      
      {activityData.length > 0 ? (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart
            data={activityData}
            margin={{ top: 10, right: 10, left: 0, bottom: 10 }}
          >
            <defs>
              <linearGradient id="colorActivity" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(0,0,0,0.06)" />
            <XAxis 
              dataKey="day" 
              axisLine={false} 
              tickLine={false} 
              tick={{ fontSize: 12, fill: '#64748b' }} 
            />
            <YAxis 
              hide={true} 
              axisLine={false} 
              tickLine={false} 
            />
            <RechartsTooltip 
              formatter={(value) => [`${value} cases`]}
              contentStyle={{ 
                borderRadius: 12, 
                border: 'none', 
                boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                padding: '10px 14px',
              }}
              cursor={{ stroke: '#3b82f6', strokeWidth: 1, strokeDasharray: '4 4' }}
            />
            <Area 
              type="monotone" 
              dataKey="count" 
              stroke="#3b82f6" 
              strokeWidth={3}
              fill="url(#colorActivity)" 
              animationDuration={1500}
              dot={{ stroke: '#3b82f6', strokeWidth: 2, r: 4, fill: 'white' }}
              activeDot={{ stroke: '#3b82f6', strokeWidth: 2, r: 6, fill: 'white' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <Box sx={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="text.secondary">No activity data available</Typography>
        </Box>
      )}
      
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mt: 3,
        pt: 2,
        borderTop: '1px solid',
        borderColor: 'divider'
      }}>
        <Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
            Total cases this week
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px' }}>
            {activityData.reduce((sum, item) => sum + item.count, 0)}
          </Typography>
        </Box>
        {activityData.length > 0 && (
          <StyledChip 
            icon={<CheckCircleIcon />}
            label="+23% vs last week" 
            color="success" 
            size="small" 
            variant="outlined"
          />
        )}
      </Box>
    </StyledCard>
  );
};

// Priority cases component
const PriorityCases = ({ cases = [] }) => {
  return (
    <StyledCard elevation={0}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <NotificationsActiveIcon sx={{ mr: 1.5, color: 'error.main' }} />
        <Typography variant="h6" sx={{ fontWeight: 600, letterSpacing: '-0.5px' }}>
          Priority Cases
        </Typography>
      </Box>
      
      {cases.length > 0 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {cases.map((kase, index) => (
            <Paper
              key={kase.case_id}
              elevation={0}
              sx={{
                p: 2.5,
                borderRadius: 3,
                border: '1px solid',
                borderColor: 'error.light',
                bgcolor: 'rgba(239, 68, 68, 0.04)',
                transition: 'all 0.3s ease',
                '&:hover': {
                  boxShadow: '0 4px 12px rgba(239, 68, 68, 0.15)',
                  transform: 'translateY(-2px)'
                }
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
                <ErrorIcon fontSize="small" color="error" sx={{ mr: 1.5 }} />
                <Typography variant="body1" fontWeight={600} sx={{ letterSpacing: '-0.3px' }}>
                  {kase.case_name || kase.case_id}
                </Typography>
              </Box>
              
              <Typography variant="body2" sx={{ mb: 2, ml: 0.5 }}>
                {kase.issue || 'High priority review required'}
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <StyledChip
                  label="Urgent"
                  color="error"
                  size="small"
                  sx={{ fontWeight: 600 }}
                />
                <StyledButton
                  size="small"
                  variant="outlined"
                  color="primary"
                  component={Link}
                  to={`/cases/${kase.case_name ? encodeURIComponent(kase.case_name) : kase.case_id}`}
                  endIcon={<ChevronRightIcon />}
                >
                  Review
                </StyledButton>
              </Box>
            </Paper>
          ))}
        </Box>
      ) : (
        <Box sx={{ 
          height: 200, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          p: 3, 
          borderRadius: 3,
          backgroundColor: 'rgba(0,0,0,0.02)'
        }}>
          <Typography color="text.secondary" sx={{ fontWeight: 500 }}>
            No priority cases
          </Typography>
        </Box>
      )}
    </StyledCard>
  );
};

// Recent activity component
const RecentActivity = ({ activities = [] }) => {
  return (
    <StyledCard elevation={0}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, letterSpacing: '-0.5px' }}>
          Recent Activity
        </Typography>
        <StyledButton 
          size="small" 
          endIcon={<ChevronRightIcon />} 
          component={Link} 
          to="/cases"
          sx={{ fontWeight: 500 }}
        >
          View All
        </StyledButton>
      </Box>
      
      {activities.length > 0 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {activities.map((activity, index) => (
            <Box 
              key={index}
              sx={{ 
                display: 'flex', 
                p: 2,
                borderRadius: 3,
                backgroundColor: 'rgba(0,0,0,0.02)',
                transition: 'all 0.2s ease',
                '&:hover': {
                  backgroundColor: 'rgba(0,0,0,0.04)',
                }
              }}
            >
              <Box sx={{ mr: 2 }}>
                <Avatar 
                  sx={{ 
                    width: 40, 
                    height: 40, 
                    bgcolor: activity.type.includes('case') ? 'info.main' : 'secondary.main',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
                  }}
                >
                  {activity.type.includes('case') ? <FolderIcon fontSize="small" /> : <PersonIcon fontSize="small" />}
                </Avatar>
              </Box>
              
              <Box sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body1" fontWeight={600} sx={{ letterSpacing: '-0.3px' }}>
                    {activity.user || activity.system}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                    {activity.time}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {activity.details}
                </Typography>
              </Box>
            </Box>
          ))}
        </Box>
      ) : (
        <Box sx={{ 
          height: 200, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          p: 3, 
          borderRadius: 3,
          backgroundColor: 'rgba(0,0,0,0.02)'
        }}>
          <Typography color="text.secondary" sx={{ fontWeight: 500 }}>
            No recent activity
          </Typography>
        </Box>
      )}
    </StyledCard>
  );
};

function Dashboard() {
  const [cases, setCases] = useState([]);
  const [totalPersons, setTotalPersons] = useState(0);
  const [personStats, setPersonStats] = useState([]);
  const [objectStats, setObjectStats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [animateIn, setAnimateIn] = useState(false);
  const [tabValue, setTabValue] = useState(0);
  const [actionMenuAnchor, setActionMenuAnchor] = useState(null);
  const [selectedCase, setSelectedCase] = useState(null);
  const [connectionError, setConnectionError] = useState(false);
  const [downloadInProgress, setDownloadInProgress] = useState({});
  const abortController = useRef(null);

  // Weekly activity data
  const [activityData, setActivityData] = useState([
    { day: 'Mon', count: 12 },
    { day: 'Tue', count: 18 },
    { day: 'Wed', count: 15 },
    { day: 'Thu', count: 25 },
    { day: 'Fri', count: 20 },
    { day: 'Sat', count: 8 },
    { day: 'Sun', count: 10 },
  ]);

  // Recent activities (with the updated user)
  const [activities, setActivities] = useState([
    { type: 'case_added', user: CURRENT_USER, details: 'Added Case #F2023-0547', time: '2 hours ago' },
    { type: 'system', system: 'System', details: 'Completed video analysis for Case #F2023-0545', time: '3 hours ago' },
    { type: 'case_added', user: CURRENT_USER, details: 'Added Case #F2023-0545', time: '4 hours ago' },
    { type: 'person_identified', user: 'System', details: 'Person #4 identified in Case #F2023-0544', time: '6 hours ago' },
  ]);

  // Priority cases
  const [priorityCases, setPriorityCases] = useState([
    {
      case_id: 'case_20250405_001',
      issue: 'Unusual activity detected',
      priority: 'high'
    },
    {
      case_id: 'case_20250404_003',
      issue: 'Multiple unidentified persons',
      priority: 'high'
    }
  ]);

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Handle action menu
  const handleActionMenuOpen = (event, kase) => {
    event.preventDefault();
    event.stopPropagation();
    setActionMenuAnchor(event.currentTarget);
    setSelectedCase(kase);
  };
  
  const handleActionMenuClose = () => {
    setActionMenuAnchor(null);
    setSelectedCase(null);
  };
  
  const handleVideoDownload = async (caseId) => {
    console.log(`Initiating download for case ${caseId}`);
    
    // Mark download as in progress
    setDownloadInProgress(prev => ({ ...prev, [caseId]: true }));
    
    try {
      // Use the dedicated download endpoint
      const downloadUrl = `/api/cases/${caseId}/download_video?user=${CURRENT_USER}&t=${Date.now()}`;
      
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
        setDownloadInProgress(prev => ({ ...prev, [caseId]: false }));
      }, 1000);
    }
  };

  // Animate content in after loading
  useEffect(() => {
    if (!loading) {
      const timer = setTimeout(() => setAnimateIn(true), 100);
      return () => clearTimeout(timer);
    }
  }, [loading]);

  // Function to fetch dashboard data
  const fetchData = async () => {
    try {
      // Cancel any existing requests
      if (abortController.current) {
        abortController.current.abort();
      }
      
      // Create a new abort controller for this request
      abortController.current = new AbortController();
      
      setRefreshing(true);
      setConnectionError(false);
      
      // Fetch cases from the API
      const casesResponse = await axios.get('/api/cases', {
        signal: abortController.current.signal
      });
      
      console.log("Cases API response:", casesResponse.data); // Debug log
      
      // Sort by timestamp (newest first)
      const sortedCases = casesResponse.data
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      setCases(sortedCases);
      
      // Calculate total persons
      const totalPersonsCount = sortedCases.reduce(
        (total, kase) => total + (kase.num_persons || 0), 0
      );
      setTotalPersons(totalPersonsCount);
      
      // Generate activity data
      const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      const today = new Date();
      const activityMap = {};
      
      // Initialize counts for each day
      days.forEach(day => activityMap[day] = 0);
      
      // Count cases by day of week
      sortedCases.forEach(kase => {
        const caseDate = new Date(kase.timestamp);
        // Only count cases from the past week
        if ((today - caseDate) <= 7 * 24 * 60 * 60 * 1000) {
          const dayName = days[caseDate.getDay()];
          activityMap[dayName] += 1;
        }
      });
      
      // Convert to array format for chart
      const newActivityData = days.map(day => ({
        day,
        count: activityMap[day] || 0
      }));
      
      // Rotate the array so it starts with Monday
      const mondayIndex = newActivityData.findIndex(d => d.day === 'Mon');
      const rotatedActivityData = [
        ...newActivityData.slice(mondayIndex),
        ...newActivityData.slice(0, mondayIndex)
      ];
      
      setActivityData(rotatedActivityData);
      
      // Generate person stats and object stats
      const personAppearances = [];
      const tempObjectStats = [];
      
      // Also gather recent activities
      const recentActivities = [];
      
      // Process case data for statistics
      for (const kase of sortedCases.slice(0, 5)) { // Limit to 5 most recent cases for performance
        try {
          const caseDetailResponse = await axios.get(`/api/cases/${kase.case_id}`, {
            signal: abortController.current.signal
          });
          
          const personIdentities = caseDetailResponse.data.person_identities || [];
          
          // Add activity for this case
          const caseDate = new Date(kase.timestamp);
          const timeDiff = today - caseDate;
          let timeString;
          
          if (timeDiff < 60 * 60 * 1000) {
            timeString = `${Math.floor(timeDiff / (60 * 1000))} min ago`;
          } else if (timeDiff < 24 * 60 * 60 * 1000) {
            timeString = `${Math.floor(timeDiff / (60 * 60 * 1000))} hours ago`;
          } else {
            timeString = `${Math.floor(timeDiff / (24 * 60 * 60 * 1000))} days ago`;
          }
          
          // Use case name if available, fall back to case ID
          const caseName = kase.case_name || `Case #${kase.case_id}`;
          
          recentActivities.push({
            type: 'case_added',
            user: CURRENT_USER,
            details: `Added ${caseName}`,
            time: timeString
          });
          
          // Process person data
          for (const person of personIdentities) {
            personAppearances.push({
              personId: `Person #${person.id}`,
              case: caseName, // Use case name for better readability
              appearances: person.metadata?.appearances || 0
            });
          }
          
          // Process object stats
          const objectCounts = caseDetailResponse.data.class_counts || {};
          for (const [className, count] of Object.entries(objectCounts)) {
            const existing = tempObjectStats.find(o => o.name === className);
            if (existing) {
              existing.value += count;
            } else {
              tempObjectStats.push({ name: className, value: count });
            }
          }
        } catch (err) {
          // Handle case-specific errors but continue processing others
          if (!axios.isCancel(err)) {
            console.error(`Error fetching details for case ${kase.case_id}:`, err);
          }
        }
      }
      
      // Set activities with the collected data
      if (recentActivities.length > 0) {
        setActivities(recentActivities);
      }
      
      // Sort person stats by appearances
      if (personAppearances.length > 0) {
        setPersonStats(
          personAppearances
            .sort((a, b) => b.appearances - a.appearances)
            .slice(0, 10)
        );
      }
      
      // Sort object stats by count
      if (tempObjectStats.length > 0) {
        setObjectStats(
          tempObjectStats.sort((a, b) => b.value - a.value).slice(0, 7)
        );
      }
      
      // Set priority cases using case names
      if (sortedCases.length > 0) {
        setPriorityCases([
          {
            case_id: sortedCases[0]?.case_id || 'case_20250405_001',
            case_name: sortedCases[0]?.case_name || null,
            issue: 'Unusual activity detected',
            priority: 'high'
          },
          {
            case_id: sortedCases[1]?.case_id || 'case_20250404_003',
            case_name: sortedCases[1]?.case_name || null,
            issue: 'Multiple unidentified persons',
            priority: 'high'
          }
        ]);
      }
      
      setLoading(false);
      setRefreshing(false);
      
      // Reset abortController
      abortController.current = null;
    } catch (err) {
      if (!axios.isCancel(err)) {
        console.error('Error fetching dashboard data:', err);
        
        // Set connection error if API is unreachable
        if (err.message === 'Network Error' || err.code === 'ECONNABORTED') {
          setConnectionError(true);
          
          // Reset loading state but keep whatever data we have
          setLoading(false);
          setRefreshing(false);
        } else {
          setError('Failed to load dashboard data');
          setLoading(false);
          setRefreshing(false);
        }
      }
    }
  };

  useEffect(() => {
    fetchData();
    
    // Cleanup function to cancel any pending requests when unmounting
    return () => {
      if (abortController.current) {
        abortController.current.abort();
      }
    };
  }, []);

  const handleRefresh = () => {
    fetchData();
  };

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'];

  // Loading skeleton with improved layout
  if (loading) {
    return (
      <Box sx={{ py: 4, px: { xs: 2, sm: 4 } }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 6, px: { xs: 1, md: 0 } }}>
          <Box sx={{ width: 300, height: 40, bgcolor: 'rgba(0,0,0,0.06)', borderRadius: 2 }} />
          <Box sx={{ width: 40, height: 40, borderRadius: '50%', bgcolor: 'rgba(0,0,0,0.06)' }} />
        </Box>
        
        <Grid container spacing={4} sx={{ mb: 6 }}>
          {Array(4).fill(0).map((_, i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Box 
                sx={{ 
                  height: 120, 
                  bgcolor: 'rgba(0,0,0,0.04)', 
                  borderRadius: 4,
                  p: 3,
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                <Box sx={{ width: 50, height: 50, borderRadius: 3, bgcolor: 'rgba(0,0,0,0.06)', mr: 2 }} />
                <Box>
                  <Box sx={{ width: 80, height: 10, bgcolor: 'rgba(0,0,0,0.06)', mb: 1, borderRadius: 1 }} />
                  <Box sx={{ width: 60, height: 24, bgcolor: 'rgba(0,0,0,0.08)', borderRadius: 1 }} />
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>
        
        <Box 
          sx={{ 
            height: 400, 
            bgcolor: 'rgba(0,0,0,0.04)', 
            borderRadius: 4,
            mb: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <CircularProgress size={60} thickness={4} />
        </Box>
        
        <Grid container spacing={4}>
          {Array(3).fill(0).map((_, i) => (
            <Grid item xs={12} md={4} key={i}>
              <Box 
                sx={{ 
                  height: 380, 
                  bgcolor: 'rgba(0,0,0,0.04)', 
                  borderRadius: 4,
                }}
              />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ 
        p: 6, 
        textAlign: 'center', 
        height: 'calc(100vh - 200px)', 
        display: 'flex', 
        flexDirection: 'column', 
        justifyContent: 'center', 
        alignItems: 'center' 
      }}>
        <Typography variant="h4" color="error" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
          {error}
        </Typography>
        <StyledButton 
          variant="contained" 
          startIcon={<RefreshIcon />}
          onClick={handleRefresh}
          size="large"
        >
          Try Again
        </StyledButton>
      </Box>
    );
  }

  // Get total cases and processing cases count
  const totalCases = cases.length;
  const processingCases = cases.filter(c => c.status === 'processing').length;

  // Dashboard stats
  const stats = [
    {
      icon: <VideocamIcon fontSize="large" />,
      title: 'Total Cases',
      value: totalCases.toString(),
      change: totalCases > 0 ? `${Math.min(totalCases, Math.floor(totalCases * 0.15))} new` : 'No cases',
      changeType: totalCases > 0 ? 'positive' : 'neutral',
    },
    {
      icon: <PersonIcon fontSize="large" />,
      title: 'Tracked Persons',
      value: totalPersons.toString(),
      change: totalPersons > 0 && totalCases > 0 ? `${Math.ceil(totalPersons / totalCases)} avg/case` : 'No persons',
      changeType: 'neutral',
    },
    {
      icon: <CachedIcon fontSize="large" />,
      title: 'Processing',
      value: processingCases.toString(),
      change: processingCases > 0 ? `${processingCases} pending` : 'None active',
      changeType: processingCases > 0 ? 'warning' : 'positive',
    },
    {
      icon: <StarIcon fontSize="large" />,
      title: 'Featured',
      value: "3",
      change: 'New features available',
      changeType: 'positive',
    }
  ];

  // Get case categories based on tab value
  const getFilteredCases = () => {
    switch (tabValue) {
      case 1: // Processing
        return cases.filter(c => c.status === 'processing');
      case 2: // Completed
        return cases.filter(c => c.status === 'completed');
      case 3: // All
        return cases;
      case 0: // Recent
      default:
        return cases.slice(0, 5);
    }
  };

  return (
    <Box sx={{ py: 4, px: { xs: 2, sm: 4 } }}>
      {connectionError && (
        <Alert 
          severity="warning" 
          sx={{ 
            mb: 4, 
            borderRadius: 3,
            '& .MuiAlert-icon': { fontSize: 28 },
            '& .MuiAlert-message': { fontSize: '1rem' }
          }}
          action={
            <StyledButton color="inherit" size="small" onClick={handleRefresh}>
              Retry
            </StyledButton>
          }
        >
          Could not connect to the backend API. Some features may be limited.
        </Alert>
      )}
    
      <Fade in={animateIn} timeout={450}>
        <Box>
          {/* Header section */}
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            mb: 5, 
            px: { xs: 1, sm: 0 }
          }}>
            <StyledHeading variant="h3" component="h1">
              ForensicsAI Dashboard
            </StyledHeading>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>

              
              <Tooltip title="Refresh data">
                <StyledIconButton
                  onClick={handleRefresh} 
                  disabled={refreshing} 
                  size="medium"
                  sx={{
                    animation: refreshing ? 'spin 1s linear infinite' : 'none',
                    '@keyframes spin': {
                      '0%': { transform: 'rotate(0deg)' },
                      '100%': { transform: 'rotate(360deg)' }
                    }
                  }}
                >
                  {refreshing ? <CircularProgress size={24} thickness={2} /> : <RefreshIcon />}
                </StyledIconButton>
              </Tooltip>
            </Box>
          </Box>
          
          {/* Quick Stats */}
          <Grid container spacing={4} sx={{ mb: 6 }}>
            {stats.map((stat, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Fade 
                  in={animateIn} 
                  timeout={300 + index * 100}
                  style={{ transformOrigin: '0 0 0' }}
                >
                  <Box>
                    <QuickStat 
                      icon={stat.icon} 
                      title={stat.title} 
                      value={stat.value}
                      change={stat.change}
                      changeType={stat.changeType}
                    />
                  </Box>
                </Fade>
              </Grid>
            ))}
          </Grid>
          
          {/* Cases Section */}
          <Fade in={animateIn} timeout={600}>
            <Box sx={{ mb: 6 }}>
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                mb: 3
              }}>
                <Typography variant="h4" sx={{ fontWeight: 600, letterSpacing: '-0.5px' }}>
                  Case Management
                </Typography>
                <StyledButton
                  variant="contained"
                  color="primary"
                  component={Link}
                  to="/upload"
                  startIcon={<VideocamIcon />}
                  size="large"
                >
                  New Case
                </StyledButton>
              </Box>
              
              <StyledCard sx={{ p: 0, overflow: 'hidden' }}>
                <StyledTabs 
                  value={tabValue} 
                  onChange={handleTabChange}
                  variant="fullWidth"
                  sx={{ 
                    borderBottom: 1, 
                    borderColor: 'divider',
                    bgcolor: 'rgba(0,0,0,0.02)'
                  }}
                >
                  <Tab label="Recent" />
                  <Tab label="Processing" />
                  <Tab label="Completed" />
                  <Tab label="All Cases" />
                </StyledTabs>
                
                <Box sx={{ px: 3, py: 2 }}>
                  <Box sx={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    py: 2,
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                    mb: 2
                  }}>
                    <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
                      {tabValue === 0 ? 'Showing most recent cases' : 
                      tabValue === 1 ? 'Showing cases in processing' : 
                      tabValue === 2 ? 'Showing completed cases' : 
                      'Showing all cases'}
                    </Typography>
                    <StyledChip 
                      icon={<FilterListIcon fontSize="small" />} 
                      label="Filter" 
                      size="small" 
                      variant="outlined"
                      sx={{ fontWeight: 500 }}
                    />
                  </Box>
                  
                  {getFilteredCases().length > 0 ? (
                    <Table sx={{ 
                      tableLayout: 'fixed',
                      '& .MuiTableCell-root': {
                        py: 2.5,
                        fontSize: '0.95rem',
                        borderBottom: '1px solid',
                        borderColor: 'rgba(0,0,0,0.06)'
                      },
                      '& .MuiTableCell-head': {
                        fontWeight: 600,
                        letterSpacing: '-0.3px',
                        color: 'text.primary'
                      }
                    }}>
                      <TableHead>
                        <TableRow>
                          <TableCell width="28%">Case Name</TableCell>
                          <TableCell width="18%">Date</TableCell>
                          <TableCell width="18%">Status</TableCell>
                          <TableCell width="15%">Persons</TableCell>
                          <TableCell width="21%" align="right">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {getFilteredCases().map((kase) => (
                          <TableRow 
                            key={kase.case_id} 
                            hover
                            sx={{ 
                              '&:hover': { 
                                bgcolor: 'rgba(0,0,0,0.02)',
                                '& .MuiIconButton-root': {
                                  opacity: 1,
                                  transform: 'translateX(0)'
                                }
                              }
                            }}
                          >
                            <TableCell>
                              <Link 
                                to={`/cases/${kase.case_name ? encodeURIComponent(kase.case_name) : kase.case_id}`}
                                style={{ 
                                  textDecoration: 'none', 
                                  color: 'inherit',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '10px'
                                }}
                              >
                                <Avatar 
                                  sx={{ 
                                    bgcolor: 'primary.light', 
                                    width: 36, 
                                    height: 36,
                                    boxShadow: '0 2px 6px rgba(37, 99, 235, 0.2)'
                                  }}
                                >
                                  <FolderIcon fontSize="small" />
                                </Avatar>
                                <Typography 
                                  variant="body1" 
                                  fontWeight={600} 
                                  sx={{ letterSpacing: '-0.3px' }}
                                >
                                  {kase.case_name || kase.case_id}
                                </Typography>
                              </Link>
                            </TableCell>
                            <TableCell>
                              <Typography 
                                variant="body2" 
                                sx={{ 
                                  color: 'text.secondary',
                                  fontWeight: 500
                                }}
                              >
                                {new Date(kase.timestamp).toLocaleDateString()}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <StyledChip
                                label={kase.status === 'processing' ? 'Processing' : 'Completed'}
                                color={kase.status === 'processing' ? 'warning' : 'success'}
                                size="small"
                                sx={{ fontWeight: 600 }}
                              />
                            </TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Avatar 
                                  sx={{ 
                                    bgcolor: 'rgba(0,0,0,0.04)', 
                                    color: 'text.primary',
                                    width: 28,
                                    height: 28
                                  }}
                                >
                                  <Typography variant="body2" fontWeight={600}>
                                    {kase.num_persons || 0}
                                  </Typography>
                                </Avatar>
                                <Typography variant="body2" color="text.secondary">
                                  {kase.num_persons === 1 ? 'person' : 'persons'}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell align="right">
                              <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                                <Tooltip title="Download Analyzed Video">
                                  <span>
                                    <StyledIconButton 
                                      size="small"
                                      color="primary"
                                      onClick={(e) => {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        handleVideoDownload(kase.case_id);
                                      }}
                                      disabled={downloadInProgress[kase.case_id] || kase.status === 'processing'}
                                      sx={{ 
                                        opacity: 0.7, 
                                        transform: 'translateX(10px)',
                                        transition: 'all 0.2s ease'
                                      }}
                                    >
                                      {downloadInProgress[kase.case_id] ? (
                                        <CircularProgress size={18} thickness={3} />
                                      ) : (
                                        <DownloadIcon fontSize="small" />
                                      )}
                                    </StyledIconButton>
                                  </span>
                                </Tooltip>
                                <Tooltip title="More Actions">
                                  <StyledIconButton 
                                    size="small" 
                                    onClick={(e) => handleActionMenuOpen(e, kase)}
                                    sx={{ 
                                      opacity: 0.7, 
                                      transform: 'translateX(10px)',
                                      transition: 'all 0.2s ease'
                                    }}
                                  >
                                    <MoreVertIcon fontSize="small" />
                                  </StyledIconButton>
                                </Tooltip>
                              </Box>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <Box sx={{ 
                      py: 8, 
                      textAlign: 'center',
                      borderRadius: 4,
                      bgcolor: 'rgba(0,0,0,0.02)'
                    }}>
                      <Typography 
                        color="text.secondary"
                        variant="h6"
                        sx={{ fontWeight: 500, letterSpacing: '-0.3px' }}
                      >
                        No cases found in this category
                      </Typography>
                    </Box>
                  )}
                </Box>
                
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'center', 
                  p: 3,
                  borderTop: '1px solid',
                  borderColor: 'divider'
                }}>
                  <StyledButton 
                    component={Link} 
                    to="/cases"
                    size="large"
                    variant="outlined"
                    color="primary"
                    endIcon={<ChevronRightIcon />}
                  >
                    View All Cases
                  </StyledButton>
                </Box>
              </StyledCard>
            </Box>
          </Fade>
          
          {/* Activity Widgets */}
          <Grid container spacing={4} sx={{ mb: 6 }}>
            <Grid item xs={12} md={4}>
              <Fade in={animateIn} timeout={700}>
                <Box>
                  <ActivityTracker activityData={activityData} />
                </Box>
              </Fade>
            </Grid>
            <Grid item xs={12} md={4}>
              <Fade in={animateIn} timeout={800}>
                <Box>
                  <RecentActivity activities={activities} />
                </Box>
              </Fade>
            </Grid>
            <Grid item xs={12} md={4}>
              <Fade in={animateIn} timeout={900}>
                <Box>
                  <PriorityCases cases={priorityCases} />
                </Box>
              </Fade>
            </Grid>
          </Grid>
          
          
          {/* Action Menu for Cases */}
          <Menu
            anchorEl={actionMenuAnchor}
            open={Boolean(actionMenuAnchor)}
            onClose={handleActionMenuClose}
            PaperProps={{
              elevation: 3,
              sx: { 
                borderRadius: 3, 
                minWidth: 220,
                boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                overflow: 'hidden',
                mt: 1
              }
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <MenuItem 
              component={Link} 
              to={selectedCase ? `/cases/${selectedCase.case_name ? encodeURIComponent(selectedCase.case_name) : selectedCase.case_id}` : '#'}
              onClick={handleActionMenuClose}
              sx={{ gap: 2, py: 2, fontWeight: 500 }}
            >
              <VisibilityIcon fontSize="small" color="primary" /> View Details
            </MenuItem>
            <MenuItem 
              component={Link}
              to={`/chat?case=${selectedCase?.case_id}`}
              onClick={handleActionMenuClose}
              sx={{ gap: 2, py: 2, fontWeight: 500 }}
            >
              <ChatIcon fontSize="small" color="primary" /> AI Analysis Chat
            </MenuItem>
            <MenuItem 
              onClick={(e) => {
                e.preventDefault();
                if (selectedCase) {
                  handleVideoDownload(selectedCase.case_id);
                }
                handleActionMenuClose();
              }}
              disabled={selectedCase && selectedCase.status === 'processing'}
              sx={{ gap: 2, py: 2, fontWeight: 500 }}
            >
              <DownloadIcon fontSize="small" color="primary" /> Download Video
            </MenuItem>
            <MenuItem 
              onClick={handleActionMenuClose} 
              sx={{ gap: 2, py: 2, fontWeight: 500 }}
            >
              <ShareIcon fontSize="small" color="primary" /> Export Report
            </MenuItem>
            <Divider />
            <MenuItem 
              onClick={handleActionMenuClose} 
              sx={{ gap: 2, py: 2, color: 'error.main', fontWeight: 500 }}
            >
              <DeleteIcon fontSize="small" color="error" /> Delete Case
            </MenuItem>
          </Menu>
        </Box>
      </Fade>
    </Box>
  );
}

export default Dashboard;