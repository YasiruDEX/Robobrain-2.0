/**
 * Draw annotations on an image canvas
 * @param {string} imageSrc - Base64 or URL of the image
 * @param {Object} annotations - Object containing points, boxes, or trajectories
 * @returns {Promise<string>} - Base64 encoded annotated image
 */
export const annotateImage = (imageSrc, annotations) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      
      // Draw the original image
      ctx.drawImage(img, 0, 0);
      
      const { points, boxes, trajectories } = annotations;
      
      // Draw points
      if (points && points.length > 0) {
        points.forEach((point, idx) => {
          const [x, y] = point;
          // Red circle with white outline
          ctx.beginPath();
          ctx.arc(x, y, 12, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgb(255, 0, 0)';
          ctx.fill();
          ctx.strokeStyle = 'rgb(255, 255, 255)';
          ctx.lineWidth = 2;
          ctx.stroke();
          
          // Add point number
          ctx.fillStyle = 'white';
          ctx.font = 'bold 12px Arial';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText((idx + 1).toString(), x, y);
        });
      }
      
      // Draw bounding boxes
      if (boxes && boxes.length > 0) {
        boxes.forEach((box, idx) => {
          const [x1, y1, x2, y2] = box;
          
          // Green semi-transparent fill
          ctx.fillStyle = 'rgba(0, 255, 0, 0.15)';
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
          
          // Green border
          ctx.strokeStyle = 'rgb(0, 255, 0)';
          ctx.lineWidth = 4;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          
          // Label background
          const label = `Object ${idx + 1}`;
          ctx.font = 'bold 14px Arial';
          const textMetrics = ctx.measureText(label);
          const textWidth = textMetrics.width;
          
          ctx.fillStyle = 'rgb(0, 255, 0)';
          ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
          
          // Label text
          ctx.fillStyle = 'black';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(label, x1 + 5, y1 - 12);
        });
      }
      
      // Draw trajectories
      if (trajectories && trajectories.length > 0) {
        trajectories.forEach((trajectory) => {
          if (trajectory.length < 2) return;
          
          // Draw lines with white outline
          ctx.lineWidth = 8;
          ctx.strokeStyle = 'rgb(255, 255, 255)';
          ctx.beginPath();
          ctx.moveTo(trajectory[0][0], trajectory[0][1]);
          for (let i = 1; i < trajectory.length; i++) {
            ctx.lineTo(trajectory[i][0], trajectory[i][1]);
          }
          ctx.stroke();
          
          // Draw orange line on top
          ctx.lineWidth = 6;
          ctx.strokeStyle = 'rgb(255, 140, 0)';
          ctx.beginPath();
          ctx.moveTo(trajectory[0][0], trajectory[0][1]);
          for (let i = 1; i < trajectory.length; i++) {
            ctx.lineTo(trajectory[i][0], trajectory[i][1]);
          }
          ctx.stroke();
          
          // Draw waypoint circles
          trajectory.forEach((point, idx) => {
            const [x, y] = point;
            
            if (idx === 0) {
              // Start point - green
              ctx.beginPath();
              ctx.arc(x, y, 10, 0, 2 * Math.PI);
              ctx.fillStyle = 'rgb(0, 255, 0)';
              ctx.fill();
              ctx.strokeStyle = 'rgb(255, 255, 255)';
              ctx.lineWidth = 2;
              ctx.stroke();
            } else if (idx === trajectory.length - 1) {
              // End point - red
              ctx.beginPath();
              ctx.arc(x, y, 12, 0, 2 * Math.PI);
              ctx.fillStyle = 'rgb(255, 0, 0)';
              ctx.fill();
              ctx.strokeStyle = 'rgb(255, 255, 255)';
              ctx.lineWidth = 2;
              ctx.stroke();
            } else {
              // Intermediate points - orange
              ctx.beginPath();
              ctx.arc(x, y, 8, 0, 2 * Math.PI);
              ctx.fillStyle = 'rgb(255, 140, 0)';
              ctx.fill();
              ctx.strokeStyle = 'rgb(255, 255, 255)';
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          });
        });
      }
      
      // Convert canvas to base64
      resolve(canvas.toDataURL('image/jpeg', 0.95));
    };
    
    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };
    
    img.src = imageSrc;
  });
};

/**
 * Parse coordinates from model response text
 * @param {string} text - Response text containing coordinates
 * @param {string} task - Task type (pointing, affordance, trajectory, grounding)
 * @returns {Object} - Parsed annotations object
 */
export const parseCoordinates = (text, task) => {
  const annotations = {
    points: null,
    boxes: null,
    trajectories: null
  };
  
  if (!text) return annotations;
  
  if (task === 'trajectory') {
    // Extract coordinates from various formats: (x,y), [x,y], or just x,y pairs
    const pattern = /[\[(]?\s*(\d+)\s*,\s*(\d+)\s*[\])]?/g;
    const matches = [...text.matchAll(pattern)];
    if (matches.length > 0) {
      annotations.trajectories = [matches.map(m => [parseInt(m[1]), parseInt(m[2])])];
    }
  } else if (task === 'pointing') {
    // Extract points from format (x, y) or [x, y]
    const pattern = /[\[(]\s*(\d+)\s*,\s*(\d+)\s*[\])]]/g;
    const matches = [...text.matchAll(pattern)];
    if (matches.length > 0) {
      annotations.points = matches.map(m => [parseInt(m[1]), parseInt(m[2])]);
    }
  } else if (task === 'affordance' || task === 'grounding') {
    // Extract bounding box [x1, y1, x2, y2]
    const pattern = /\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]/g;
    const matches = [...text.matchAll(pattern)];
    if (matches.length > 0) {
      annotations.boxes = matches.map(m => [
        parseInt(parseFloat(m[1])),
        parseInt(parseFloat(m[2])),
        parseInt(parseFloat(m[3])),
        parseInt(parseFloat(m[4]))
      ]);
    }
  }
  
  return annotations;
};
