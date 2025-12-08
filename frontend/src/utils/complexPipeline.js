/**
 * Complex Pipeline Service
 * 
 * This module handles decomposition of complex queries into sub-tasks
 * and orchestrates sequential execution through RoboBrain.
 */

import { sendMessage } from '../api';
import { annotateImage, parseCoordinates } from './imageAnnotation';

// Task descriptions for the decomposition prompt
const ROBOBRAIN_TASKS = {
    general: {
        description: "General visual question answering - describe scenes, count objects, identify colors, explain content",
        output: "Text description or answer",
        examples: ["What is in this image?", "How many objects are there?", "What color is the car?"]
    },
    grounding: {
        description: "Locate and find objects in the image, returns bounding box coordinates [x1,y1,x2,y2]",
        output: "Bounding box coordinates",
        examples: ["Where is the cup?", "Find the apple", "Locate the chair"]
    },
    affordance: {
        description: "Detect graspable areas and manipulation affordances, returns bounding box of graspable region",
        output: "Bounding box of graspable area",
        examples: ["Where to grasp the cup?", "How to pick up this object?", "Find graspable area"]
    },
    trajectory: {
        description: "Plan motion paths and trajectories, returns sequence of waypoints [(x1,y1), (x2,y2), ...]",
        output: "List of waypoint coordinates",
        examples: ["Plan path to the cup", "How to reach the object?", "Generate trajectory to target"]
    },
    pointing: {
        description: "Point to specific locations or identify what's at a point, returns single (x,y) coordinate",
        output: "Single point coordinate",
        examples: ["Point to the button", "Click on the handle", "What is at (100,200)?"]
    }
};

/**
 * Build task descriptions string for the system prompt
 */
const buildTaskDescriptions = () => {
    return Object.entries(ROBOBRAIN_TASKS)
        .map(([task, info]) => `- ${task}: ${info.description} (Output: ${info.output})`)
        .join('\n');
};

/**
 * Decompose a complex query into a pipeline of sub-tasks using Groq API
 * This calls the backend endpoint that uses Groq for decomposition
 * 
 * Returns: { pipeline: Array, fallback: boolean, error: string|null }
 */
export const decomposeQuery = async (query) => {
    const response = await fetch(`${import.meta.env.VITE_API_URL || ''}/api/decompose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    });

    if (!response.ok) {
        throw new Error('Failed to decompose query');
    }

    const data = await response.json();

    // Log for debugging
    if (data.fallback) {
        console.warn('[Pipeline] Decomposition fell back to general task:', data.error || data.message);
    } else {
        console.log('[Pipeline] Successfully decomposed into', data.pipeline?.length, 'steps');
    }

    return {
        pipeline: data.pipeline,
        fallback: data.fallback || false,
        error: data.error || null,
        message: data.message || null
    };
};

/**
 * Format previous steps as dialog context for the next step
 */
const formatContextAsDialog = (previousSteps) => {
    if (!previousSteps || previousSteps.length === 0) return '';

    let context = '[Previous conversation context]\n';

    previousSteps.forEach((step, idx) => {
        context += `User (Step ${idx + 1} - ${step.task}): ${step.prompt}\n`;
        context += `Assistant: ${step.answer || step.result || 'No response'}\n`;

        // Include coordinate information if available
        if (step.thinking) {
            context += `(Coordinates detected: ${step.thinking})\n`;
        }
    });

    context += '[Current query]\n';
    return context;
};

/**
 * Execute a pipeline of tasks sequentially
 * 
 * @param {string} sessionId - Current session ID
 * @param {Array} pipeline - Array of pipeline steps
 * @param {Object} imageRef - Reference to uploaded image
 * @param {Function} onStepStart - Callback when step starts
 * @param {Function} onStepComplete - Callback when step completes
 * @returns {Object} - Combined results with all annotations
 */
export const executePipeline = async (
    sessionId,
    pipeline,
    imageRef,
    currentImagePreview,
    onStepStart,
    onStepComplete
) => {
    const results = [];
    const allAnnotations = {
        points: [],
        boxes: [],
        trajectories: []
    };

    for (let i = 0; i < pipeline.length; i++) {
        const step = pipeline[i];

        // Notify step started
        if (onStepStart) {
            onStepStart(i, step);
        }

        try {
            // Build prompt with context from previous steps if use_previous is true
            let prompt = step.prompt;
            if (step.use_previous && results.length > 0) {
                const context = formatContextAsDialog(results);
                prompt = `${context}${step.prompt}`;
            }

            // Send the request to backend
            const response = await sendMessage(sessionId, prompt, {
                image: i === 0 ? imageRef : null, // Only send image on first request
                task: step.task,
                enableThinking: false,
            });

            // Parse coordinates from response
            const textToUse = response.answer || response.thinking || '';
            const annotations = parseCoordinates(textToUse, step.task);

            // Merge annotations
            if (annotations.points) {
                allAnnotations.points.push(...annotations.points);
            }
            if (annotations.boxes) {
                allAnnotations.boxes.push(...annotations.boxes);
            }
            if (annotations.trajectories) {
                allAnnotations.trajectories.push(...annotations.trajectories);
            }

            // Store result
            const stepResult = {
                step: i + 1,
                task: step.task,
                prompt: step.prompt,
                description: step.description,
                answer: response.answer,
                thinking: response.thinking,
                outputImage: response.output_image,
                annotations,
                success: true
            };

            results.push(stepResult);

            // Notify step completed
            if (onStepComplete) {
                onStepComplete(i, stepResult);
            }

        } catch (error) {
            console.error(`Pipeline step ${i + 1} failed:`, error);

            const stepResult = {
                step: i + 1,
                task: step.task,
                prompt: step.prompt,
                description: step.description,
                error: error.message,
                success: false
            };

            results.push(stepResult);

            if (onStepComplete) {
                onStepComplete(i, stepResult);
            }
        }
    }

    // Combine all annotations into a final image
    let finalOutputImage = null;
    const hasAnnotations =
        allAnnotations.points.length > 0 ||
        allAnnotations.boxes.length > 0 ||
        allAnnotations.trajectories.length > 0;

    if (hasAnnotations && currentImagePreview) {
        try {
            finalOutputImage = await annotateImage(currentImagePreview, {
                points: allAnnotations.points.length > 0 ? allAnnotations.points : null,
                boxes: allAnnotations.boxes.length > 0 ? allAnnotations.boxes : null,
                trajectories: allAnnotations.trajectories.length > 0 ? allAnnotations.trajectories : null,
            });
        } catch (err) {
            console.error('Failed to create combined annotation:', err);
        }
    }

    // Build summary answer
    const summaryParts = results.map(r => {
        if (r.success) {
            return `**Step ${r.step} (${r.task})**: ${r.description}\n${r.answer || 'Completed'}`;
        } else {
            return `**Step ${r.step} (${r.task})**: Failed - ${r.error}`;
        }
    });

    return {
        results,
        allAnnotations,
        finalOutputImage,
        summary: summaryParts.join('\n\n'),
        success: results.every(r => r.success)
    };
};

/**
 * Check if a query is complex enough to need decomposition
 * Simple heuristics to detect complex queries
 */
export const isComplexQuery = (query) => {
    const complexIndicators = [
        'and then',
        'after that',
        'first',
        'next',
        'finally',
        'pick up',
        'move to',
        'grab',
        'navigate to',
        'go to',
        'place',
        'put',
        'open',
        'close',
        'turn',
    ];

    const lowerQuery = query.toLowerCase();
    return complexIndicators.some(indicator => lowerQuery.includes(indicator));
};

export { ROBOBRAIN_TASKS };
