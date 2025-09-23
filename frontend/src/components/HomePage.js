import React, { useState, useEffect } from 'react';
import { 
  Box, Typography, Button, Grid, Paper, Fade, Grow, 
  Container, Divider, Card, CardContent, Avatar, 
  IconButton, Chip, useTheme, useMediaQuery, Stack
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { Link } from 'react-router-dom';

// Icons
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import VideocamIcon from '@mui/icons-material/Videocam';
import PersonIcon from '@mui/icons-material/Person';
import TimelineIcon from '@mui/icons-material/Timeline';
import BarChartIcon from '@mui/icons-material/BarChart';
import ChatIcon from '@mui/icons-material/Chat';
import SecurityIcon from '@mui/icons-material/Security';
import DescriptionIcon from '@mui/icons-material/Description';
import ShieldIcon from '@mui/icons-material/Shield';
import GavelIcon from '@mui/icons-material/Gavel';
import AssignmentIcon from '@mui/icons-material/Assignment';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import LocalPoliceIcon from '@mui/icons-material/LocalPolice';
import FingerprintIcon from '@mui/icons-material/Fingerprint';
import ArticleIcon from '@mui/icons-material/Article';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PsychologyIcon from '@mui/icons-material/Psychology';
import LockIcon from '@mui/icons-material/Lock';
import FolderIcon from '@mui/icons-material/Folder';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';

// Current date and user info (updated with the exact values provided)
const CURRENT_DATE = '2025-04-06 10:53:33';
const CURRENT_USER = 'aaravgoel0';

// Styled Components
const HeroSection = styled(Box)(({ theme }) => ({
  position: 'relative',
  minHeight: '500px',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  overflow: 'hidden',
  padding: theme.spacing(6, 4),
  backgroundColor: theme.palette.mode === 'dark' ? '#10192d' : '#f8fafc',
  [theme.breakpoints.up('md')]: {
    padding: theme.spacing(8, 6),
  },
}));

const FeatureCard = styled(Card)(({ theme }) => ({
  height: '100%',
  borderRadius: 8,
  transition: 'all 0.2s ease',
  boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
  overflow: 'hidden',
  border: '1px solid rgba(0,0,0,0.08)',
}));

const StatCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: 8,
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
  alignItems: 'center',
  boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
}));

const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: 4,
  padding: '10px 24px',
  textTransform: 'none',
  fontWeight: 600,
  fontSize: '0.95rem',
  boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
}));

const FeatureIcon = styled(Avatar)(({ theme }) => ({
  width: 52,
  height: 52,
  backgroundColor: theme.palette.primary.main,
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  marginBottom: theme.spacing(2),
}));

// Badge styling
const Badge = styled(Box)(({ theme }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  padding: '4px 12px',
  borderRadius: 4,
  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
  marginBottom: theme.spacing(3),
  fontWeight: 600,
  fontSize: '0.85rem',
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontWeight: 700,
  letterSpacing: '-0.02em',
  marginBottom: theme.spacing(1),
}));

const HomePage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isMedium = useMediaQuery(theme.breakpoints.down('md'));
  const [animateHero, setAnimateHero] = useState(false);
  const [animateFeatures, setAnimateFeatures] = useState(false);

  useEffect(() => {
    // Trigger animations after component mount
    setAnimateHero(true);
    
    // Stagger the animations
    const timer = setTimeout(() => {
      setAnimateFeatures(true);
    }, 400);
    
    return () => clearTimeout(timer);
  }, []);

  // Statistics for the stats section
  const stats = [
    { label: 'Analysis Accuracy', icon: <VerifiedUserIcon fontSize="large" color="primary" /> },
    { label: 'Evidence Secured', icon: <ShieldIcon fontSize="large" color="primary" /> },
    { label: 'Cases Processed', icon: <AssignmentIcon fontSize="large" color="primary" /> },
    { label: 'Agency Partners', icon: <LocalPoliceIcon fontSize="large" color="primary" /> },
  ];

  // Feature data with verified correct links
  const features = [
    {
      title: 'Secure Evidence Management',
      description: 'End-to-end encrypted storage with comprehensive chain of custody tracking for all digital evidence.',
      icon: <LockIcon fontSize="large" />,
      color: theme.palette.primary.main,
      link: '/cases'
    },
    {
      title: 'Advanced Person Identification',
      description: 'High-precision facial recognition and person tracking using state-of-the-art AI algorithms.',
      icon: <PersonIcon fontSize="large" />,
      color: theme.palette.primary.dark,
      link: '/cases'
    },
    {
      title: 'Forensic Timeline Analysis',
      description: 'Chronological reconstruction of events with precise timestamp correlation across multiple evidence sources.',
      icon: <TimelineIcon fontSize="large" />,
      color: theme.palette.primary.main,
      link: '/cases'
    },
    {
      title: 'AI Investigation Assistant',
      description: 'Conversational AI tool for evidence analysis, pattern identification, and investigative assistance.',
      icon: <SmartToyIcon fontSize="large" />,
      color: theme.palette.primary.dark,
      link: '/chat'
    },
    {
      title: 'Court-Ready Reports',
      description: 'Generate comprehensive, court-admissible documentation of all findings with one click.',
      icon: <ArticleIcon fontSize="large" />,
      color: theme.palette.primary.main,
      link: '/reports'
    },
    {
      title: 'Compliance & Audit Trails',
      description: 'Comprehensive logging and documentation for legal compliance and evidence admissibility.',
      icon: <GavelIcon fontSize="large" />,
      color: theme.palette.primary.dark,
      link: '/cases'
    },
  ];

  return (
    <Box sx={{ overflowX: 'hidden' }}>
      {/* Hero Section */}
      <HeroSection>
        <Container maxWidth="lg">
          <Fade in={animateHero} timeout={1000}>
            <Box>
              <Badge>
                <ShieldIcon fontSize="small" sx={{ mr: 1 }} />
                ForensicsAI Platform • OFFICIAL USE ONLY
              </Badge>
              
              <Typography 
                variant="h2" 
                component="h1" 
                sx={{ 
                  fontWeight: 700, 
                  fontSize: { xs: '2rem', sm: '2.5rem', md: '3rem' },
                  lineHeight: 1.2,
                  mb: 2,
                  letterSpacing: '-0.02em' 
                }}
              >
                Advanced Video Analysis for<br/>
                Law Enforcement & Forensic Investigators
              </Typography>
              
              <Typography 
                variant="h6" 
                color="text.secondary" 
                sx={{ 
                  maxWidth: 600, 
                  mb: 4, 
                  fontWeight: 400,
                  lineHeight: 1.5,
                }}
              >
                Secure, court-admissible video evidence analysis with chain-of-custody tracking, person identification, and comprehensive reporting.
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              </Box>
            </Box>
          </Fade>
        </Container>
        
        {/* Subtle decorative element */}
        <Box sx={{ 
          position: 'absolute', 
          width: '100%',
          height: '100%',
          top: 0,
          left: 0,
          opacity: 0.03,
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000...`,
          zIndex: 0
        }} />
      </HeroSection>

      {/* Main Features Section */}
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 5 }}>
          <SectionTitle variant="h4">
            Comprehensive Forensic Analysis Tools
          </SectionTitle>
          <Typography 
            variant="body1" 
            color="text.secondary" 
            sx={{ maxWidth: 800 }}
          >
            Our platform provides law enforcement with secure, court-admissible evidence processing and analysis capabilities.
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          {features.map((feature, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Grow in={animateFeatures} timeout={300 + index * 100}>
                <Box>
                  <FeatureCard>
                    <CardContent sx={{ p: 3 }}>
                      <FeatureIcon sx={{ bgcolor: feature.color }}>
                        {feature.icon}
                      </FeatureIcon>
                      
                      <Typography 
                        variant="h6" 
                        component="h3"
                        sx={{ 
                          fontWeight: 600, 
                          mb: 1.5,
                        }}
                      >
                        {feature.title}
                      </Typography>
                      
                      <Typography 
                        variant="body2" 
                        color="text.secondary"
                        sx={{ 
                          mb: 2,
                          lineHeight: 1.6
                        }}
                      >
                        {feature.description}
                      </Typography>
                      
                      <Button 
                        component={Link}
                        to={feature.link}
                        endIcon={<ArrowForwardIcon />}
                        size="small"
                        sx={{ 
                          textTransform: 'none',
                          fontWeight: 500
                        }}
                      >
                        Access module
                      </Button>
                    </CardContent>
                  </FeatureCard>
                </Box>
              </Grow>
            </Grid>
          ))}
        </Grid>
      </Container>
      
      {/* Statistics Section */}
      <Container maxWidth="lg" sx={{ py: 5 }}>
        <Box sx={{ mb: 4 }}>
          <SectionTitle variant="h4">
            ForensicsAI Platform Statistics
          </SectionTitle>
          <Typography 
            variant="body1" 
            color="text.secondary" 
          >
            Trusted by law enforcement agencies and forensic laboratories worldwide
          </Typography>
        </Box>
        
        <Fade in={animateFeatures} timeout={800}>
          <Grid container spacing={3}>
            {stats.map((stat, index) => (
              <Grid item xs={6} md={3} key={index}>
                <StatCard elevation={0}>
                  <Box sx={{ mb: 1 }}>
                    {stat.icon}
                  </Box>
                  <Typography 
                    variant="h4" 
                    sx={{ 
                      fontWeight: 700, 
                      mb: 1,
                    }}
                  >
                    {stat.value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {stat.label}
                  </Typography>
                </StatCard>
              </Grid>
            ))}
          </Grid>
        </Fade>
      </Container>
      
      {/* Quick Access Section */}
      <Container maxWidth="lg" sx={{ pb: 8, pt: 2 }}>
        <SectionTitle variant="h5" sx={{ mb: 3 }}>
          Quick Access
        </SectionTitle>
        
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={600}>
              <Paper 
                component={Link} 
                to="/dashboard"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <BarChartIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  Dashboard
                </Typography>
              </Paper>
            </Fade>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={800}>
              <Paper 
                component={Link} 
                to="/cases"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <FolderIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  Cases
                </Typography>
              </Paper>
            </Fade>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={1000}>
              <Paper 
                component={Link} 
                to="/upload"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <VideocamIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  Upload
                </Typography>
              </Paper>
            </Fade>
          </Grid>
          
          <Grid item xs={6} sm={3}>
            <Fade in={animateFeatures} timeout={1200}>
              <Paper 
                component={Link} 
                to="/reports"
                elevation={0} 
                sx={{ 
                  p: 2, 
                  borderRadius: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  textDecoration: 'none',
                  color: 'text.primary',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: theme.palette.primary.main,
                    bgcolor: 'action.hover'
                  }
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: theme.palette.primary.main, 
                    width: 40, 
                    height: 40,
                  }}
                >
                  <PictureAsPdfIcon />
                </Avatar>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  Reports
                </Typography>
              </Paper>
            </Fade>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default HomePage;