/**
 * Resize image if dimensions exceed maxDimension while maintaining aspect ratio
 * @param {string} imageSrc - Base64 or URL of the image
 * @param {number} maxDimension - Maximum width or height (default: 2000)
 * @returns {Promise<Object>} - Object containing resized image data and scale factor
 */
export const resizeImageIfNeeded = (imageSrc, maxDimension = 2000) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      const { width, height } = img;
      const maxDim = Math.max(width, height);

      // If image is within limits, return original
      if (maxDim <= maxDimension) {
        resolve({
          dataUrl: imageSrc,
          scaleFactor: 1,
          originalWidth: width,
          originalHeight: height,
          newWidth: width,
          newHeight: height
        });
        return;
      }

      // Calculate new dimensions maintaining aspect ratio
      const scaleFactor = maxDimension / maxDim;
      const newWidth = Math.round(width * scaleFactor);
      const newHeight = Math.round(height * scaleFactor);

      // Create canvas and resize
      const canvas = document.createElement('canvas');
      canvas.width = newWidth;
      canvas.height = newHeight;
      const ctx = canvas.getContext('2d');

      // Use high quality image smoothing
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      ctx.drawImage(img, 0, 0, newWidth, newHeight);

      resolve({
        dataUrl: canvas.toDataURL('image/jpeg', 0.92),
        scaleFactor,
        originalWidth: width,
        originalHeight: height,
        newWidth,
        newHeight
      });
    };

    img.onerror = () => {
      reject(new Error('Failed to load image for resizing'));
    };

    img.src = imageSrc;
  });
};

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

  // Pattern for 4-value bounding box: [x1, y1, x2, y2]
  const boxPattern = /\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]/g;

  // Pattern for 2-value point in various formats: (x, y) or [x, y] or [(x, y)]
  const pointPattern = /[\[(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\])]/g;

  // Pattern for trajectory array: [(x1,y1), (x2,y2), ...]
  const trajectoryPattern = /\[\s*\([\d\s,]+\)(?:\s*,\s*\([\d\s,]+\))*\s*\]/g;

  if (task === 'trajectory') {
    // For trajectory, extract all (x,y) pairs
    const allPoints = [];
    let match;

    // Reset pattern
    const coordPattern = /\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)/g;
    while ((match = coordPattern.exec(text)) !== null) {
      allPoints.push([parseInt(parseFloat(match[1])), parseInt(parseFloat(match[2]))]);
    }

    // Also try [x, y] format if no parentheses found
    if (allPoints.length === 0) {
      const bracketPattern = /\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]/g;
      while ((match = bracketPattern.exec(text)) !== null) {
        // Make sure it's not a 4-value box
        const fullMatch = match[0];
        const commaCount = (fullMatch.match(/,/g) || []).length;
        if (commaCount === 1) {
          allPoints.push([parseInt(parseFloat(match[1])), parseInt(parseFloat(match[2]))]);
        }
      }
    }

    if (allPoints.length > 0) {
      annotations.trajectories = [allPoints];
    }
  } else if (task === 'pointing') {
    // Extract single points from format (x, y) or [x, y]
    const points = [];
    let match;

    // First try (x, y) format
    const parenPattern = /\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)/g;
    while ((match = parenPattern.exec(text)) !== null) {
      points.push([parseInt(parseFloat(match[1])), parseInt(parseFloat(match[2]))]);
    }

    // Also try [x, y] format (2 values only)
    if (points.length === 0) {
      const bracketPattern = /\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]/g;
      while ((match = bracketPattern.exec(text)) !== null) {
        // Check it's only 2 values (not a bounding box)
        const beforeBracket = text.substring(0, match.index);
        const afterBracket = text.substring(match.index + match[0].length);
        points.push([parseInt(parseFloat(match[1])), parseInt(parseFloat(match[2]))]);
      }
    }

    if (points.length > 0) {
      annotations.points = points;
    }
  } else if (task === 'affordance' || task === 'grounding') {
    // First try to find 4-value bounding boxes [x1, y1, x2, y2]
    const boxes = [];
    let match;

    while ((match = boxPattern.exec(text)) !== null) {
      boxes.push([
        parseInt(parseFloat(match[1])),
        parseInt(parseFloat(match[2])),
        parseInt(parseFloat(match[3])),
        parseInt(parseFloat(match[4]))
      ]);
    }

    if (boxes.length > 0) {
      annotations.boxes = boxes;
    } else {
      // If no 4-value boxes found, look for 2-value points
      // Format: [(x, y)] or (x, y) - treat as center point and create small box
      const points = [];

      // Try (x, y) format first
      const parenPattern = /\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)/g;
      while ((match = parenPattern.exec(text)) !== null) {
        points.push([parseInt(parseFloat(match[1])), parseInt(parseFloat(match[2]))]);
      }

      // Try [x, y] format (2 values only)
      if (points.length === 0) {
        // Look for simple 2-value arrays
        const simpleArrayPattern = /\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]/g;
        let simpleMatch;
        while ((simpleMatch = simpleArrayPattern.exec(text)) !== null) {
          // Count commas to ensure it's a 2-value array
          const matchStr = simpleMatch[0];
          const commaCount = (matchStr.match(/,/g) || []).length;
          if (commaCount === 1) {
            points.push([parseInt(parseFloat(simpleMatch[1])), parseInt(parseFloat(simpleMatch[2]))]);
          }
        }
      }

      // Convert points to small boxes for visualization (create 40px box around point)
      if (points.length > 0) {
        annotations.boxes = points.map(([x, y]) => [
          Math.max(0, x - 20),
          Math.max(0, y - 20),
          x + 20,
          y + 20
        ]);
        // Also store original points for reference
        annotations.points = points;
      }
    }
  }

  console.log(`[parseCoordinates] Task: ${task}, Found:`, annotations);
  return annotations;
};

