import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Typography, Box, Grid, Paper, Button, Chip, Divider, CircularProgress, 
         Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
         ScatterChart, Scatter, ZAxis } from 'recharts';
import axios from 'axios';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import PersonIcon from '@mui/icons-material/Person';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import TimelineIcon from '@mui/icons-material/Timeline';

function PersonDetail() {
  const { caseId, personId } = useParams();
  const [personData, setPersonData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchPersonData() {
      try {
        setLoading(true);
        
        // Get person details
        const response = await axios.get(`/api/cases/${caseId}/persons/${personId}`);
        setPersonData(response.data);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching person details:', err);
        setError('Failed to load person data');
        setLoading(false);
      }
    }
    
    fetchPersonData();
  }, [caseId, personId]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  if (!personData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="70vh">
        <Typography>Person not found</Typography>
      </Box>
    );
  }

  // Format appearance data for timeline chart
  const appearanceData = (personData.appearances || []).map(app => ({
    time: app.timestamp,
    frame: app.frame,
    confidence: app.confidence || 0.5
  }));

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Person #{personId} - Case {caseId}
        </Typography>
        <Box>
          <Button 
            component={Link} 
            to={`/cases/${caseId}`} 
            variant="outlined" 
            startIcon={<ArrowBackIcon />}
            sx={{ mr: 2 }}
          >
            Back to Case
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
      
      {/* Person Summary */}
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            {personData.thumbnail ? (
              <img 
                src={personData.thumbnail} 
                alt={`Person #${personId}`}
                style={{ width: '100%', height: 'auto', maxHeight: '300px', objectFit: 'contain', border: '1px solid #ddd' }}
              />
            ) : (
              <Box 
                sx={{ 
                  width: '100%', 
                  height: 300, 
                  display: 'flex', 
                  justifyContent: 'center', 
                  alignItems: 'center',
                  bgcolor: 'grey.200',
                  border: '1px solid #ddd'
                }}
              >
                <PersonIcon sx={{ fontSize: 80, color: 'grey.400' }} />
              </Box>
            )}
          </Grid>
          
          <Grid item xs={12} md={8}>
            <Typography variant="h5" gutterBottom>
              Person #{personId}
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Appearances: {personData.metadata?.appearances || personData.appearances?.length || 0}
              </Typography>
              
              {personData.metadata?.first_seen_time && (
                <Typography variant="body2" gutterBottom>
                  First seen: Frame {personData.metadata.first_seen_frame} at {new Date(personData.metadata.first_seen_time * 1000).toLocaleTimeString()}
                </Typography>
              )}
              
              {personData.metadata?.last_seen_time && (
                <Typography variant="body2" gutterBottom>
                  Last seen: Frame {personData.metadata.last_seen_frame} at {new Date(personData.metadata.last_seen_time * 1000).toLocaleTimeString()}
                </Typography>
              )}
              
              {personData.position_3d && (
                <Typography variant="body2" gutterBottom>
                  <Chip 
                    icon={<ViewInArIcon />} 
                    label="3D Position Available" 
                    color="success"
                    variant="outlined"
                    sx={{ mt: 1 }}
                  />
                </Typography>
              )}
            </Box>
            
            {/* Display attributes if available */}
            {personData.metadata?.attributes && Object.keys(personData.metadata.attributes).length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>Attributes</Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {Object.entries(personData.metadata.attributes).map(([key, value]) => {
                    // Handle different attribute formats
                    let displayValue = value;
                    if (Array.isArray(value) || typeof value === 'object') {
                      try {
                        displayValue = JSON.stringify(value).slice(0, 15) + '...';
                      } catch {
                        displayValue = 'Complex Value';
                      }
                    }
                    
                    return (
                      <Chip 
                        key={key}
                        label={`${key}: ${displayValue}`}
                        variant="outlined"
                        size="small"
                      />
                    );
                  })}
                </Box>
              </Box>
            )}
          </Grid>
        </Grid>
      </Paper>
      
      {/* Appearance Timeline */}
      {appearanceData.length > 0 && (
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>Appearance Timeline</Typography>
          
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={appearanceData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                label={{ value: 'Time (seconds)', position: 'insideBottomRight', offset: -10 }}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <YAxis 
                label={{ value: 'Confidence', angle: -90, position: 'insideLeft' }}
                domain={[0, 1]}
              />
              <Tooltip 
                formatter={(value, name, props) => [`${value.toFixed(2)}`, name]}
                labelFormatter={(value) => `Time: ${value.toFixed(2)}s, Frame: ${appearanceData.find(d => d.time === value)?.frame}`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="confidence" 
                stroke="#8884d8" 
                activeDot={{ r: 8 }}
                name="Detection Confidence"
              />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      )}
      
      {/* 3D Trajectory */}
      {personData.trajectory_3d && personData.trajectory_3d.length > 0 && (
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>3D Trajectory</Typography>
          
          {/* Transform 3D data for visualization */}
          {(() => {
            // Prepare data for 2D visualization (top-down view)
            const trajectoryData = personData.trajectory_3d.map((point, index) => ({
              x: point[0], // X coordinate
              z: point[2], // Z coordinate (using as Y for top-down view)
              y: point[1], // Y coordinate (height)
              index: index
            }));
            
            return (
              <Box>
                <Typography variant="subtitle2" gutterBottom color="text.secondary">
                  Top-down view (X-Z plane)
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart
                    margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                  >
                    <CartesianGrid />
                    <XAxis 
                      type="number" 
                      dataKey="x" 
                      name="X Position" 
                      label={{ value: 'X Coordinate', position: 'insideBottomRight', offset: -10 }}
                    />
                    <YAxis 
                      type="number" 
                      dataKey="z" 
                      name="Z Position" 
                      label={{ value: 'Z Coordinate', angle: -90, position: 'insideLeft' }}
                    />
                    <ZAxis 
                      type="number" 
                      dataKey="y" 
                      name="Height" 
                      range={[20, 100]} 
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(value, name) => [`${value.toFixed(2)}`, name]}
                      labelFormatter={(value) => `Point ${value + 1}`}
                    />
                    <Scatter 
                      name="Person Position" 
                      data={trajectoryData} 
                      fill="#8884d8" 
                      line={{ stroke: '#8884d8', strokeWidth: 2 }}
                      lineType="fitting"
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </Box>
            );
          })()}
        </Paper>
      )}
      
      {/* Appearance Details Table */}
      {personData.appearances && personData.appearances.length > 0 && (
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>Appearance Details</Typography>
          
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Frame</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Bounding Box</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {personData.appearances.slice(0, 20).map((appearance, index) => (
                  <TableRow key={index}>
                    <TableCell>{appearance.frame}</TableCell>
                    <TableCell>{appearance.timestamp ? appearance.timestamp.toFixed(2) + 's' : 'N/A'}</TableCell>
                    <TableCell>{appearance.confidence ? (appearance.confidence * 100).toFixed(1) + '%' : 'N/A'}</TableCell>
                    <TableCell>
                      {appearance.box ? (
                        `[${appearance.box.map(v => Math.round(v)).join(', ')}]`
                      ) : 'N/A'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          {personData.appearances.length > 20 && (
            <Typography variant="body2" sx={{ mt: 2, textAlign: 'center', color: 'text.secondary' }}>
              Showing 20 of {personData.appearances.length} appearances
            </Typography>
          )}
        </Paper>
      )}
    </Box>
  );
}

export default PersonDetail;