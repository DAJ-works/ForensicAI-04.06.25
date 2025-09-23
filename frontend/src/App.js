import React, { useState, useEffect, useMemo } from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { 
  Container, AppBar, Toolbar, Typography, Box, CssBaseline, ThemeProvider, 
  createTheme, useMediaQuery, IconButton, Drawer, List, ListItem, 
  ListItemIcon, ListItemText, Divider, Avatar, Fade, Tooltip, 
  Menu, MenuItem, Switch, Badge, Backdrop, CircularProgress,
  Snackbar, Alert, LinearProgress, Button
} from '@mui/material';
import { styled } from '@mui/material/styles';
import Dashboard from './components/Dashboard';
import CaseList from './components/CaseList';
import CaseDetail from './components/CaseDetail';
import PersonDetail from './components/PersonDetail';
import TimelineView from './components/TimelineView';
import Search from './components/Search';
import UploadForm from './components/UploadForm';
import CaseChat from './components/CaseChat'; // Import the new CaseChat component


// Icons
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import FolderIcon from '@mui/icons-material/Folder';
import SearchIcon from '@mui/icons-material/Search';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SettingsIcon from '@mui/icons-material/Settings';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import BarChartIcon from '@mui/icons-material/BarChart';
import LogoutIcon from '@mui/icons-material/Logout';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import SpeedIcon from '@mui/icons-material/Speed';
import ReportIcon from '@mui/icons-material/Report';
import ChatIcon from '@mui/icons-material/Chat'; // Add Chat icon
import VideoPlayer from './components/VideoPlayer';
import HomePage from './components/HomePage';

//NEW
import ChatDashboard from './components/ChatDashboard';

// Default placeholder image for when images fail to load
const placeholderImg = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"%3E%3Crect width="100" height="100" fill="%23f0f0f0"/%3E%3Ctext x="50%25" y="50%25" font-family="Arial" font-size="14" text-anchor="middle" dominant-baseline="middle" fill="%23a0a0a0"%3EVP%3C/text%3E%3C/svg%3E';

// Create themes with dark mode support
const createAppTheme = (mode) => createTheme({
  palette: {
    mode,
    primary: {
      main: mode === 'light' ? '#2563eb' : '#60a5fa', // Blue
      light: mode === 'light' ? '#60a5fa' : '#93c5fd',
      dark: mode === 'light' ? '#1e40af' : '#3b82f6',
    },
    secondary: {
      main: mode === 'light' ? '#9333ea' : '#c4b5fd', // Purple
      light: mode === 'light' ? '#c084fc' : '#ddd6fe',
      dark: mode === 'light' ? '#6b21a8' : '#a78bfa',
    },
    background: {
      default: mode === 'light' ? '#f8fafc' : '#0f172a',
      paper: mode === 'light' ? '#ffffff' : '#1e293b',
    },
    text: {
      primary: mode === 'light' ? '#0f172a' : '#f1f5f9',
      secondary: mode === 'light' ? '#64748b' : '#cbd5e1',
    },
    success: {
      main: '#10b981',
      light: '#34d399',
      dark: '#059669',
    },
    warning: {
      main: '#f59e0b',
      light: '#fbbf24',
      dark: '#d97706',
    },
    error: {
      main: '#ef4444',
      light: '#f87171',
      dark: '#dc2626',
    },
    info: {
      main: '#0ea5e9',
      light: '#38bdf8',
      dark: '#0284c7',
    }
  },
  typography: {
    fontFamily: [
      'Inter',
      'Roboto',
      'system-ui',
      '-apple-system',
      'sans-serif',
    ].join(','),
    h4: {
      fontWeight: 700,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 10,
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: 'none',
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          padding: '6px 16px',
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
          borderRadius: '0.75rem',
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: '0.75rem',
        }
      }
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: '0.5rem',
        }
      }
    },
    MuiContainer: {
      styleOverrides: {
        root: {
          paddingLeft: '12px',
          paddingRight: '12px',
          '@media (min-width: 600px)': {
            paddingLeft: '16px',
            paddingRight: '16px',
          },
        }
      }
    }
  },
});

// Styled drawer
const drawerWidth = 250;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })(
  ({ theme, open }) => ({
    flexGrow: 1,
    transition: theme.transitions.create(['margin', 'width'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    marginLeft: 0,
    ...(open && {
      transition: theme.transitions.create(['margin', 'width'], {
        easing: theme.transitions.easing.easeOut,
        duration: theme.transitions.duration.enteringScreen,
      }),
      [theme.breakpoints.up('md')]: {
        marginLeft: drawerWidth,
        width: `calc(100% - ${drawerWidth}px)`,
      },
    }),
  }),
);

// Date formatter utility
const formatDateTime = () => {

};

// Image with fallback component
const ImageWithFallback = ({ src, alt, ...props }) => {
  const [imgSrc, setImgSrc] = useState(src || placeholderImg);
  
  useEffect(() => {
    setImgSrc(src || placeholderImg);
  }, [src]);
  
  return (
    <img
      src={imgSrc}
      alt={alt}
      onError={() => setImgSrc(placeholderImg)}
      loading="lazy"
      {...props}
    />
  );
};

// Quick Action Button component with animation
const QuickActionButton = ({ icon, title, onClick, color = "primary" }) => {
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <Tooltip title={title} arrow>
      <IconButton 
        color={color}
        onClick={onClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        sx={{ 
          transition: 'all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
          transform: isHovered ? 'scale(1.15)' : 'scale(1)',
          '&:active': {
            transform: 'scale(0.95)'
          }
        }}
      >
        {icon}
      </IconButton>
    </Tooltip>
  );
};

// System stats component
const SystemStatus = () => {
  const [performance, setPerformance] = useState({
    cpu: 68,
    memory: 45,
    storage: 82,
  });
  
  useEffect(() => {
    const timer = setInterval(() => {
      // Simulate changing system metrics
      setPerformance({
        cpu: Math.floor(Math.random() * 30) + 60, // 60-90%
        memory: Math.floor(Math.random() * 20) + 40, // 40-60%
        storage: 82,
      });
    }, 5000);
    
    return () => clearInterval(timer);
  }, []);
  
  return 
  
};

// Main App component
function App() {
  const [useDarkMode, setUseDarkMode] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true); 
  const isMobile = useMediaQuery('(max-width:900px)');
  const [isLoading, setIsLoading] = useState(false);
  const [notifications, setNotifications] = useState(3);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const currentTime = formatDateTime();
  
  // Create theme based on dark mode setting
  const theme = useMemo(() => createAppTheme(useDarkMode ? 'dark' : 'light'), [useDarkMode]);
  
  // Settings menu state
  const [settingsAnchorEl, setSettingsAnchorEl] = useState(null);
  const settingsMenuOpen = Boolean(settingsAnchorEl);
  
  // Notification menu state
  const [notifAnchorEl, setNotifAnchorEl] = useState(null);
  const notifMenuOpen = Boolean(notifAnchorEl);
  
  // Handle settings menu
  const handleSettingsClick = (event) => {
    setSettingsAnchorEl(event.currentTarget);
  };
  
  const handleSettingsClose = () => {
    setSettingsAnchorEl(null);
  };
  
  // Handle notification menu
  const handleNotifClick = (event) => {
    setNotifAnchorEl(event.currentTarget);
    setNotifications(0); // Clear notifications when opened
  };
  
  const handleNotifClose = () => {
    setNotifAnchorEl(null);
  };
  
  // Toggle dark mode
  const handleToggleDarkMode = () => {
    setIsLoading(true);
    // Simulate a loading time for the theme switch
    setTimeout(() => {
      setUseDarkMode(!useDarkMode);
      setIsLoading(false);
      setSnackbar({
        open: true, 
        message: !useDarkMode ? 'Dark mode activated' : 'Light mode activated', 
        severity: 'success'
      });
    }, 400);
    handleSettingsClose();
  };
  
  // Close snackbar
  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') return;
    setSnackbar({ ...snackbar, open: false });
  };
  
  useEffect(() => {
    // Initial drawer state based on screen size
    if (isMobile) {
      setDrawerOpen(false);
    }
  }, [isMobile]);

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };
  
  // Mock data
  const mockNotifications = [
    { id: 1, title: "New case uploaded", description: "Case #F2023-0542 has been processed", time: "10 min ago", read: false },
    { id: 2, title: "Processing complete", description: "Video analysis completed for Case #F2023-0539", time: "1 hour ago", read: false },
    { id: 3, title: "System update available", description: "VideoProof v2.1.0 is ready to install", time: "2 hours ago", read: false },
  ];
  
  // Simulate page transition loading
  const simulatePageLoading = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 700);
  };
  
  const showSnackbarMessage = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };
  
  // NEW
  const drawerItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard', badge: 0 },
    { text: 'Cases', icon: <FolderIcon />, path: '/cases', badge: 2 },
    { text: 'Upload', icon: <CloudUploadIcon />, path: '/upload', badge: 0 },
    { text: 'AI Chat', icon: <ChatIcon />, path: '/chat', badge: 0 }, // Changed from '/cases' to '/chat'
  ];
  
  // Navigation link component with active state
  const NavLink = ({ to, icon, text, badge, onClick }) => {
    const location = useLocation();
    const isActive = location.pathname === to || 
      (to !== '/' && location.pathname.startsWith(to));
    
    return (
      <ListItem 
        button 
        component={Link} 
        to={to}
        onClick={() => {
          onClick && onClick();
          simulatePageLoading();
        }}
        sx={{
          borderRadius: '0.5rem',
          mb: 0.5,
          mx: 1,
          bgcolor: isActive ? 'rgba(37, 99, 235, 0.08)' : 'transparent',
          color: isActive ? 'primary.main' : 'inherit',
          transition: 'all 0.3s ease',
          '&:hover': {
            bgcolor: isActive ? 'rgba(37, 99, 235, 0.12)' : 'rgba(0, 0, 0, 0.04)',
            transform: 'translateX(4px)',
          },
        }}
      >
        <ListItemIcon sx={{ color: isActive ? 'primary.main' : 'inherit', minWidth: 40 }}>
          {badge ? (
            <Badge badgeContent={badge} color="error" variant="dot">
              {icon}
            </Badge>
          ) : icon}
        </ListItemIcon>
        <ListItemText primary={text} />
      </ListItem>
    );
  };
  
  const drawer = (
    <>
      
      <Divider sx={{ mx: 2, opacity: 0.6 }} />
      
      
      {/* System status panel */}
      <SystemStatus />
      
      <Divider sx={{ mx: 2, my: 1, opacity: 0.6 }} />
      
      {/* Navigation menu */}
      <Box 
        sx={{ 
          overflowY: 'auto', 
          pt: 1, 
          pb: 2, 
          height: isMobile ? 'auto' : 'calc(100vh - 340px)'
        }}
      >
        <Typography 
          variant="overline" 
          sx={{ 
            display: 'block',
            px: 3, 
            color: 'text.secondary', 
            fontWeight: 600,
            fontSize: '0.7rem',
            letterSpacing: 1,
            mb: 0.5
          }}
        >
          MAIN MENU
        </Typography>
        
        <List component="nav" disablePadding>
          {drawerItems.map((item) => (
            <NavLink 
              key={item.text}
              to={item.path}
              icon={item.icon}
              text={item.text}
              badge={item.badge}
              onClick={isMobile ? handleDrawerToggle : undefined}
            />
          ))}
        </List>
        
        <Box sx={{ mt: 2 }}>
          <Typography 
            variant="overline" 
            sx={{ 
              display: 'block',
              px: 3, 
              color: 'text.secondary', 
              fontWeight: 600,
              fontSize: '0.7rem',
              letterSpacing: 1,
              mb: 0.5
            }}
          >
          </Typography>
          
          <List component="nav" disablePadding>
        
          </List>
        </Box>
      </Box>
      
      {/* Bottom section with dark mode toggle */}
      <Box sx={{ 
        mt: 'auto', 
        p: 2, 
        borderTop: `1px solid ${theme.palette.divider}`,
      }}>
        <Box sx={{ 
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          p: 1,
          borderRadius: 2,
        }}>
          <Typography variant="body2">Dark Mode</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <LightModeIcon 
              fontSize="small" 
              color={useDarkMode ? 'disabled' : 'warning'} 
              sx={{ opacity: useDarkMode ? 0.5 : 1 }}
            />
            <Switch 
              checked={useDarkMode} 
              onChange={handleToggleDarkMode}
              size="small"
              color="primary"
            />
            <DarkModeIcon 
              fontSize="small" 
              color={useDarkMode ? 'primary' : 'disabled'} 
              sx={{ opacity: useDarkMode ? 1 : 0.5 }}
            />
          </Box>
        </Box>
      
        
     
      </Box>
    </>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ 
          display: 'flex', 
          minHeight: '100vh',
          bgcolor: theme.palette.background.default,
        }}>
          {/* Global loading indicator */}
          {isLoading && (
            <LinearProgress 
              sx={{ 
                position: 'fixed', 
                top: 0, 
                left: 0, 
                right: 0, 
                zIndex: theme.zIndex.drawer + 2,
                height: 3
              }} 
            />
          )}

          {/* Settings menu */}
          <Menu
            anchorEl={settingsAnchorEl}
            open={settingsMenuOpen}
            onClose={handleSettingsClose}
            PaperProps={{
              elevation: 3,
              sx: { 
                mt: 1.5, 
                borderRadius: 2,
                minWidth: 200,
                overflow: 'visible',
                '&:before': {
                  content: '""',
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  right: 14,
                  width: 10,
                  height: 10,
                  bgcolor: 'background.paper',
                  transform: 'translateY(-50%) rotate(45deg)',
                  zIndex: 0,
                },
              },
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <MenuItem onClick={handleToggleDarkMode} sx={{ gap: 2 }}>
              {useDarkMode ? (
                <><LightModeIcon fontSize="small" /> Light Mode</>
              ) : (
                <><DarkModeIcon fontSize="small" /> Dark Mode</>
              )}
            </MenuItem>
            <MenuItem sx={{ gap: 2 }}>
              <SpeedIcon fontSize="small" /> Performance
            </MenuItem>
            <MenuItem sx={{ gap: 2 }}>
              <SettingsIcon fontSize="small" /> Preferences
            </MenuItem>
            <Divider />
            <MenuItem sx={{ gap: 2 }}>
              <HelpOutlineIcon fontSize="small" /> Help & Support
            </MenuItem>
          </Menu>
          
          {/* Notifications menu */}
          <Menu
            anchorEl={notifAnchorEl}
            open={notifMenuOpen}
            onClose={handleNotifClose}
            PaperProps={{
              elevation: 3,
              sx: { 
                mt: 1.5, 
                borderRadius: 2,
                minWidth: 280,
                maxWidth: 320,
                overflow: 'visible',
                '&:before': {
                  content: '""',
                  display: 'block',
                  position: 'absolute',
                  top: 0,
                  right: 14,
                  width: 10,
                  height: 10,
                  bgcolor: 'background.paper',
                  transform: 'translateY(-50%) rotate(45deg)',
                  zIndex: 0,
                },
              },
            }}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
          >
            <Box sx={{ p: 2, pb: 1 }}>
              <Typography variant="subtitle1" fontWeight={600}>
                Notifications
              </Typography>
              <Typography variant="body2" color="text.secondary">
                You have {mockNotifications.filter(n => !n.read).length} unread notifications
              </Typography>
            </Box>
            <Divider />
            {mockNotifications.map(notification => (
              <MenuItem key={notification.id} sx={{ px: 2, py: 1.5 }}>
                <Box sx={{ width: '100%' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="subtitle2">
                      {notification.title}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {notification.time}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                    {notification.description}
                  </Typography>
                </Box>
              </MenuItem>
            ))}
            <Divider />
            <Box sx={{ p: 1.5, textAlign: 'center' }}>
              <Button size="small" onClick={() => showSnackbarMessage('Viewed all notifications', 'success')}>
                View All
              </Button>
            </Box>
          </Menu>
          
          {/* Drawer */}
          <Box
            component="nav"
            sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
          >
            {/* Mobile drawer - temporary */}
            {isMobile && (
              <Drawer
                variant="temporary"
                open={drawerOpen}
                onClose={handleDrawerToggle}
                ModalProps={{ keepMounted: true }}
                sx={{
                  '& .MuiDrawer-paper': { 
                    boxSizing: 'border-box', 
                    width: drawerWidth,
                    borderRight: 'none',
                    borderRadius: 0,
                  },
                  '& .MuiPaper-root': {
                    width: drawerWidth,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                  }
                }}
              >
                {drawer}
              </Drawer>
            )}
            
            {/* Desktop drawer - persistent */}
            {!isMobile && (
              <Drawer
                variant="persistent"
                open={drawerOpen}
                sx={{
                  '& .MuiDrawer-paper': { 
                    boxSizing: 'border-box', 
                    width: drawerWidth,
                    borderRight: `1px solid ${theme.palette.divider}`,
                    borderRadius: 0,
                    bgcolor: theme.palette.background.paper,
                    boxShadow: theme.palette.mode === 'light' 
                      ? '1px 0 5px 0 rgba(0,0,0,0.05)' 
                      : 'none',
                  },
                }}
              >
                {drawer}
              </Drawer>
            )}
          </Box>
          
          {/* Main Content */}
          <Main open={drawerOpen && !isMobile}>
            <Toolbar /> {/* For spacing below app bar */}
            <Container
              maxWidth={false}
              disableGutters
              sx={{ 
                pt: 2,
                pb: 4,
                px: { xs: 2, sm: 3 }, // Reduced padding to fix alignment
                height: 'calc(100vh - 64px)',
                overflow: 'auto',
              }}
            >
              <Fade in={true} timeout={450}>
                <Box sx={{ maxWidth: '100%', overflowX: 'hidden' }}>
                  <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/cases" element={<CaseList />} />
                    <Route path="/cases/:caseId" element={<CaseDetail />} />
                    <Route path="/cases/:caseId/persons/:personId" element={<PersonDetail />} />
                    <Route path="/cases/:caseId/timeline" element={<TimelineView />} />
                    <Route path="/case/:caseId/chat" element={<CaseChat />} /> {/* Added new route for chat */}
                    <Route path="/chat" element={<ChatDashboard />} /> {/* Add this new route */} //NEW
                    <Route path="/search" element={<Search />} />
                    <Route path="/upload" element={<UploadForm />} />
                    <Route path="*" element={<Dashboard />} />
                    <Route path="/cases/:caseId/video" element={<VideoPlayer />} />
                  </Routes>
                </Box>
              </Fade>
            </Container>
          </Main>
          
          <Snackbar 
            open={snackbar.open} 
            autoHideDuration={4000} 
            onClose={handleSnackbarClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            sx={{ mb: 2 }}
          >
            <Alert 
              onClose={handleSnackbarClose} 
              severity={snackbar.severity} 
              variant="filled"
              sx={{ width: '100%', boxShadow: '0 3px 10px rgba(0,0,0,0.2)' }}
            >
              {snackbar.message}
            </Alert>
          </Snackbar>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
               