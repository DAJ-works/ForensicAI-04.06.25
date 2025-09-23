import React, { useState } from 'react';
import { Typography, Box, TextField, Button, Paper, Grid, Card, CardContent, 
         CardMedia, Tabs, Tab, Chip, FormControl, InputLabel, Select, 
         MenuItem, CircularProgress } from '@mui/material';
import { Link } from 'react-router-dom';
import axios from 'axios';
import PersonIcon from '@mui/icons-material/Person';
import EventIcon from '@mui/icons-material/Event';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';

function Search() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState('person');
  const [caseFilter, setCaseFilter] = useState('all');
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cases, setCases] = useState([]);
  const [casesLoaded, setCasesLoaded] = useState(false);

  // Load available cases
  React.useEffect(() => {
    async function fetchCases() {
      try {
        const response = await axios.get('/api/cases');
        setCases(response.data);
        setCasesLoaded(true);
      } catch (err) {
        console.error('Error fetching cases:', err);
      }
    }
    
    if (!casesLoaded) {
      fetchCases();
    }
  }, [casesLoaded]);

  const handleSearch = async () => {
    if (searchQuery.trim() === '') {
      setError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/search', {
        type: searchType,
        query: searchQuery,
        case_id: caseFilter !== 'all' ? caseFilter : undefined
      });
      
      setSearchResults(response.data);
      setLoading(false);
    } catch (err) {
      console.error('Search error:', err);
      setError('Error performing search');
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Search & Filter
      </Typography>
      
      {/* Search Form */}
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={4}>
            <TextField
              label="Search Query"
              variant="outlined"
              fullWidth
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={searchType === 'person' ? "Enter person ID or attributes" : "Search event description"}
            />
          </Grid>
          
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Search Type</InputLabel>
              <Select
                value={searchType}
                label="Search Type"
                onChange={(e) => setSearchType(e.target.value)}
              >
                <MenuItem value="person">Persons</MenuItem>
                <MenuItem value="event">Events</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Case Filter</InputLabel>
              <Select
                value={caseFilter}
                label="Case Filter"
                onChange={(e) => setCaseFilter(e.target.value)}
              >
                <MenuItem value="all">All Cases</MenuItem>
                {cases.map((kase) => (
                  <MenuItem key={kase.case_id} value={kase.case_id}>
                    {kase.case_id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={2}>
            <Button 
              variant="contained" 
              color="primary" 
              fullWidth 
              onClick={handleSearch}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Search'}
            </Button>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Error Message */}
      {error && (
        <Box sx={{ mb: 2 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      )}
      
      {/* Search Results */}
      {searchResults.length > 0 ? (
        <Box>
          <Typography variant="h5" gutterBottom>
            Search Results ({searchResults.length})
          </Typography>
          
          <Grid container spacing={3}>
            {searchResults.map((result, index) => (
              <Grid item key={index} xs={12} sm={6} md={4} lg={3}>
                {result.type === 'person' ? (
                  <Card>
                    {result.thumbnail ? (
                      <CardMedia
                        component="img"
                        height="140"
                        image={result.thumbnail}
                        alt={`Person #${result.id}`}
                      />
                    ) : (
                      <Box 
                        sx={{ 
                          height: 140, 
                          bgcolor: 'grey.300', 
                          display: 'flex', 
                          justifyContent: 'center', 
                          alignItems: 'center' 
                        }}
                      >
                        <PersonIcon sx={{ fontSize: 40 }} />
                      </Box>
                    )}
                    <CardContent>
                      <Typography variant="h6" component="div" gutterBottom>
                        Person #{result.id}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Case: {result.case_id}
                      </Typography>
                      {result.metadata && (
                        <>
                          <Typography variant="body2" gutterBottom>
                            Appearances: {result.metadata.appearances || 0}
                          </Typography>
                          {result.metadata.first_seen_time && (
                            <Typography variant="body2" gutterBottom>
                              First seen: {new Date(result.metadata.first_seen_time * 1000).toLocaleTimeString()}
                            </Typography>
                          )}
                        </>
                      )}
                      <Button 
                        component={Link}
                        to={`/cases/${result.case_id}/persons/${result.id}`}
                        variant="contained"
                        color="primary"
                        size="small"
                        sx={{ mt: 1 }}
                      >
                        View Details
                      </Button>
                    </CardContent>
                  </Card>
                ) : result.type === 'event' ? (
                  <Card>
                    <Box 
                      sx={{ 
                        height: 60, 
                        bgcolor: 'primary.main', 
                        color: 'white',
                        display: 'flex', 
                        justifyContent: 'center', 
                        alignItems: 'center' 
                      }}
                    >
                      <EventIcon sx={{ mr: 1 }} />
                      <Typography variant="h6">
                        {result.event_type}
                      </Typography>
                    </Box>
                    <CardContent>
                      <Typography variant="body1" gutterBottom>
                        {result.description || "Event"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Case: {result.case_id}
                      </Typography>
                      {result.timestamp && (
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Time: {new Date(result.timestamp * 1000).toLocaleTimeString()}
                        </Typography>
                      )}
                      {result.frame && (
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Frame: {result.frame}
                        </Typography>
                      )}
                      <Button 
                        component={Link}
                        to={`/cases/${result.case_id}/timeline`}
                        variant="contained"
                        color="primary"
                        size="small"
                        sx={{ mt: 1 }}
                      >
                        View Timeline
                      </Button>
                    </CardContent>
                  </Card>
                ) : (
                  <Card>
                    <CardContent>
                      <Typography variant="h6" component="div" gutterBottom>
                        {result.type}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {JSON.stringify(result)}
                      </Typography>
                    </CardContent>
                  </Card>
                )}
              </Grid>
            ))}
          </Grid>
        </Box>
      ) : loading ? (
        <Box display="flex" justifyContent="center" my={4}>
          <CircularProgress />
        </Box>
      ) : searchQuery !== '' ? (
        <Typography variant="body1">
          No results found. Try a different search query or filter.
        </Typography>
      ) : null}
      
      {/* Search Tips */}
      <Paper elevation={2} sx={{ p: 3, mt: 4, bgcolor: 'info.light', color: 'info.contrastText' }}>
        <Typography variant="h6" gutterBottom>
          Search Tips
        </Typography>
        <Typography variant="body2" paragraph>
          • To search for persons, enter a person ID or specific attribute values
        </Typography>
        <Typography variant="body2" paragraph>
          • For event searching, use keywords like "appears", "disappears", or describe the event
        </Typography>
        <Typography variant="body2">
          • Use the case filter to narrow down results to a specific analysis case
        </Typography>
      </Paper>
    </Box>
  );
}

export default Search;