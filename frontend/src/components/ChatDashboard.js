import React, { useState, useEffect } from 'react';
import { 
  Typography, Box, Card, CardContent, CardActionArea, Grid, 
  Skeleton, Paper, Divider, Chip, Avatar, Button, 
  LinearProgress, TextField, InputAdornment
} from '@mui/material';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import SearchIcon from '@mui/icons-material/Search';
import ChatIcon from '@mui/icons-material/Chat';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import FilterListIcon from '@mui/icons-material/FilterList';

const ChatDashboard = () => {
  const [cases, setCases] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    // Fetch all cases
    const fetchCases = async () => {
      try {
        const response = await axios.get('/api/cases');
        setCases(response.data);
      } catch (error) {
        console.error('Error fetching cases:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchCases();
  }, []);

  // Filter cases based on search input
  const filteredCases = cases.filter(caseItem => 
    caseItem.case_id?.toLowerCase().includes(search.toLowerCase()) || 
    caseItem.video_path?.toLowerCase().includes(search.toLowerCase())
  );

  // Function to get the file name from a path
  const getFileName = (path) => {
    if (!path) return "Unknown file";
    return path.split('/').pop();
  };

  const handleCaseClick = (caseId) => {
    navigate(`/case/${caseId}/chat`);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" sx={{ mb: 1 }}>
          AI Chat Assistant
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Select a case to chat with AI about its analysis and findings
        </Typography>
      </Box>


      {/* Cases Grid */}
      {isLoading ? (
        <Grid container spacing={3}>
          {[1, 2, 3, 4].map((item) => (
            <Grid item xs={12} sm={6} md={4} key={item}>
              <Card elevation={1}>
                <CardContent>
                  <Skeleton variant="rectangular" height={40} sx={{ mb: 2 }} />
                  <Skeleton variant="text" />
                  <Skeleton variant="text" width="60%" />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Grid container spacing={3}>
          {filteredCases.map((caseItem) => (
            <Grid item xs={12} sm={6} md={4} key={caseItem.case_id}>
              <Card 
                elevation={1}
                sx={{ 
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4
                  }
                }}
              >
                <CardActionArea onClick={() => handleCaseClick(caseItem.case_id)}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Avatar
                        variant="rounded"
                        sx={{ 
                          bgcolor: 'primary.main', 
                          mr: 2 
                        }}
                      >
                        <InsertDriveFileIcon />
                      </Avatar>
                      <Box>
                        <Typography variant="h6" noWrap>
                          Case #{caseItem.case_id}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" noWrap>
                          {getFileName(caseItem.video_path)}
                        </Typography>
                      </Box>
                    </Box>
                    
                    <Divider sx={{ mb: 2 }} />
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Box>
                        {caseItem.num_persons > 0 && (
                          <Chip 
                            size="small" 
                            label={`${caseItem.num_persons} persons`} 
                            sx={{ mr: 1, mb: 1 }} 
                          />
                        )}
                        {caseItem.status === 'processing' && (
                          <Chip 
                            size="small" 
                            label="Processing" 
                            color="warning" 
                            sx={{ mb: 1 }} 
                          />
                        )}
                      </Box>
                      <Button
                        size="small"
                        color="primary"
                        startIcon={<ChatIcon />}
                      >
                        Chat
                      </Button>
                    </Box>
                    
                    {caseItem.status === 'processing' && (
                      <LinearProgress 
                        variant="determinate" 
                        value={caseItem.progress || 0} 
                        sx={{ mt: 1 }} 
                      />
                    )}
                  </CardContent>
                </CardActionArea>
              </Card>
            </Grid>
          ))}

          {filteredCases.length === 0 && (
            <Box sx={{ py: 8, textAlign: 'center', width: '100%' }}>
              <Typography variant="h6" color="text.secondary">
                No cases found matching your search
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Try a different search term or upload a new case
              </Typography>
            </Box>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default ChatDashboard;